"""Custom branched rollout manager for repo-local DAPO training."""

# pyright: reportMissingImports=false

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import torch
from tensordict import TensorDict

from branching_dapo.bootstrap import ensure_repo_paths
from branching_dapo.config_types import BranchAdvantageIndex, BranchingRolloutSettings
from branching_dapo.runtime_metrics import record_generation_metrics
from branching_dapo.rollout_utils import (
    LeafBatchRecord,
    PromptGroup,
    build_prompt_groups,
    build_reward_scores,
    extract_prompt_text,
    resolve_leaf_branch_metadata,
    summarize_rollout_records,
)

ensure_repo_paths()

from branching_eval.artifact_store import ArtifactStore  # noqa: E402
from branching_eval.branch_executor import BranchExecutor  # noqa: E402
from branching_eval.config_types import BranchingConfig, DecodingConfig  # noqa: E402
from branching_eval.selector_types import SelectorMode  # noqa: E402
from verl.experimental.agent_loop.agent_loop import AgentLoopManager  # noqa: E402
from verl.protocol import DataProto  # noqa: E402
from verl.utils.ray_utils import auto_await  # noqa: E402
from verl.utils.tokenizer import hf_tokenizer, normalize_token_ids, set_pad_token_id  # noqa: E402
from verl.utils.model import compute_position_id_with_mask  # noqa: E402
from vllm_client import VllmClient  # noqa: E402


@dataclass(frozen=True)
class PromptGroupGeneration:
    """Branched rollout outputs and metrics for one repeated-prompt group.

    Args:
        records: Leaf rollout records realized for one prompt group.
        metrics: Numeric group-level branching metrics.

    Returns:
        Prompt-group generation payload.
    """

    records: list[LeafBatchRecord]
    metrics: dict[str, float]


@dataclass(frozen=True)
class ArtifactBatchScope:
    """Resolved artifact paths and ids for one rollout batch.

    Args:
        run_name: Unique run label derived from the experiment name.
        batch_name: Unique batch label within one trainer process.
        run_dir: Per-run artifact directory.
        batch_dir: Per-batch artifact directory.
        run_id: Event-stream run identifier emitted into `tree_events.jsonl`.

    Returns:
        Immutable artifact scope metadata.
    """

    run_name: str
    batch_name: str
    run_dir: Path
    batch_dir: Path
    run_id: str


class InstrumentedBranchExecutor(BranchExecutor):
    """Branch executor with lightweight metric capture for rollout logging."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.candidate_pool_resolutions = 0
        self.cluster_counts: list[float] = []

    async def _resolve_candidate_pool_async(self, **kwargs):
        """Resolve one candidate pool and count the realized branch point."""

        pool = await super()._resolve_candidate_pool_async(**kwargs)
        self.candidate_pool_resolutions += 1
        return pool

    async def _resolve_selection_outcomes_async(self, *, pool):
        """Resolve selector outcomes and capture cluster-count diagnostics.

        Args:
            pool: Candidate pool whose selectors should be resolved.

        Returns:
            Selection outcomes in requested-selector order.
        """

        outcomes = await super()._resolve_selection_outcomes_async(pool=pool)
        active_selection = next((item for item in outcomes if item.selector_mode == self.active_selector), None)
        if active_selection is None or active_selection.cluster_by_candidate_id is None:
            self.cluster_counts.append(0.0)
            return outcomes
        cluster_count = float(len(set(active_selection.cluster_by_candidate_id.values())))
        self.cluster_counts.append(cluster_count)
        return outcomes

    def metrics(self) -> dict[str, float]:
        """Return numeric branching metrics captured during rollout.

        Args:
            None.

        Returns:
            Numeric branching metrics for one prompt group.
        """

        cluster_count_mean = float(np.mean(self.cluster_counts)) if self.cluster_counts else 0.0
        return {
            "branching/branch_point_count": float(self.candidate_pool_resolutions),
            "branching/cluster_count_mean": cluster_count_mean,
        }


class BranchingAgentLoopManager(AgentLoopManager):
    """Shared-branch rollout manager that returns all realized leaves per prompt."""

    def __init__(self, config, worker_group=None, rollout_resource_pool=None, reward_loop_worker_handles=None):
        super().__init__(config, worker_group, rollout_resource_pool, reward_loop_worker_handles)
        self.settings = BranchingRolloutSettings.from_config(config=self.config)
        self.settings.cache_root.mkdir(parents=True, exist_ok=True)
        self.branch_task_semaphore: asyncio.Semaphore | None = None
        self.branch_task_semaphore_loop_id: int | None = None
        self.artifact_run_scope = self._build_run_scope()
        self.artifact_run_scope.run_dir.mkdir(parents=True, exist_ok=True)
        self.batch_counter = 0
        trust_remote_code = bool(self.config.data.get("trust_remote_code", False))
        self.tokenizer = hf_tokenizer(self.model_config.path, trust_remote_code=trust_remote_code)
        set_pad_token_id(self.tokenizer)
        self.doc_counter = 0
        self.served_model_name = str(self.model_config.path)

    def _ensure_branch_task_semaphore(self) -> asyncio.Semaphore:
        """Return a shared branch semaphore bound to the current event loop.

        Args:
            None.

        Returns:
            Loop-local semaphore shared across prompt groups in one rollout batch.

        Example:
            The PPO trainer calls `generate_sequences()` via `asyncio.run(...)` on
            every step, so this helper recreates the semaphore when the event loop
            changes to avoid cross-loop binding errors on step 2+.
        """

        running_loop_id = id(asyncio.get_running_loop())
        if (
            self.branch_task_semaphore is None
            or self.branch_task_semaphore_loop_id != running_loop_id
        ):
            self.branch_task_semaphore = asyncio.Semaphore(
                self.settings.max_concurrent_branches
            )
            self.branch_task_semaphore_loop_id = running_loop_id
        return self.branch_task_semaphore

    def _build_run_scope(self) -> ArtifactBatchScope:
        """Build the unique artifact scope shared by one trainer process.

        Args:
            None.

        Returns:
            Run-level artifact scope with a unique directory and run id.
        """

        experiment_name = str(getattr(self.config.trainer, "experiment_name", "branching_dapo"))
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"{experiment_name}_{timestamp}"
        run_dir = self.settings.artifact_run_dir(run_name=run_name)
        return ArtifactBatchScope(
            run_name=run_name,
            batch_name="",
            run_dir=run_dir,
            batch_dir=run_dir,
            run_id=self.settings.sanitize_path_component(raw_name=run_name),
        )

    def _build_batch_scope(self, *, global_step: int) -> ArtifactBatchScope:
        """Build the artifact scope for one rollout batch.

        Args:
            global_step: Trainer step number forwarded by the PPO loop.

        Returns:
            Batch-level artifact scope nested under the run directory.
        """

        batch_name = f"batch_{self.batch_counter:04d}_step_{global_step:06d}"
        batch_dir = self.settings.artifact_batch_dir(
            run_name=self.artifact_run_scope.run_name,
            batch_name=batch_name,
        )
        return ArtifactBatchScope(
            run_name=self.artifact_run_scope.run_name,
            batch_name=batch_name,
            run_dir=self.artifact_run_scope.run_dir,
            batch_dir=batch_dir,
            run_id=self.settings.sanitize_path_component(
                raw_name=f"{self.artifact_run_scope.run_name}_{batch_name}"
            ),
        )

    def _build_batch_artifact_store(self, *, prompts: DataProto) -> ArtifactStore:
        """Build the artifact store for one rollout batch.

        Args:
            prompts: Prompt batch passed by the PPO trainer.

        Returns:
            Artifact store isolated to one run and one training batch.
        """

        global_step = int(prompts.meta_info.get("global_steps", -1))
        batch_scope = self._build_batch_scope(global_step=global_step)
        batch_scope.batch_dir.mkdir(parents=True, exist_ok=True)
        self.batch_counter += 1
        return ArtifactStore(
            run_dir=batch_scope.batch_dir,
            run_id=batch_scope.run_id,
        )

    async def _init_agent_loop_workers(self):
        """Skip stock agent-loop workers because branching is managed centrally.

        Args:
            None.

        Returns:
            None.
        """

        self.agent_loop_workers = []

    def _build_executor(
        self,
        *,
        prompt_text: str,
        doc_id: int,
        artifact_store: ArtifactStore,
    ) -> InstrumentedBranchExecutor:
        """Construct one branch executor bound to a rollout server.

        Args:
            prompt_text: Plain-text user prompt.
            doc_id: Stable prompt-group document id.
            artifact_store: Batch-scoped artifact store for event emission.

        Returns:
            Configured branch executor for one prompt group.
        """

        server_address = self.server_addresses[doc_id % len(self.server_addresses)]
        client = VllmClient(base_url=server_address)
        branching = BranchingConfig(
            branch_prob=self.settings.branch_prob,
            max_branch_points_per_rollout=self.settings.max_branch_points_per_rollout,
            max_concurrent_branches=self.settings.max_concurrent_branches,
            num_candidates=self.settings.num_candidates,
            branch_fanout=self.settings.branch_fanout,
            max_clusters=self.settings.max_clusters,
            candidate_span_tokens=self.settings.candidate_span_tokens,
            max_steer_tokens=self.settings.max_steer_tokens,
            steer_repetition_penalty=self.settings.steer_repetition_penalty,
            entropy_threshold=self.settings.entropy_threshold,
            entropy_profile_name=self.settings.entropy_profile_name,
        )
        decoding = DecodingConfig(
            temperature=float(self.rollout_config.temperature),
            top_p=float(self.rollout_config.top_p),
            max_gen_toks=int(self.rollout_config.response_length),
            top_logprobs=20,
            decode_chunk_tokens=min(512, int(self.rollout_config.response_length)),
        )
        selector_mode = cast(SelectorMode, self.settings.validated_selector_mode())
        return InstrumentedBranchExecutor(
            client=client,
            cluster_client=None,
            prompt_text=prompt_text,
            model_name=self.served_model_name,
            cluster_model_name=None,
            decoding=decoding,
            branching=branching,
            artifact_store=artifact_store,
            requested_selectors=(selector_mode,),
            active_selector=selector_mode,
            seed=self.settings.seed + doc_id,
            trigger_steer_enabled=self.settings.trigger_steer_enabled,
            trigger_entropy_enabled=self.settings.trigger_entropy_enabled,
            env_paths=self.settings.env_paths,
            branch_task_semaphore=self._ensure_branch_task_semaphore(),
        )

    def _tokenize_prompt(self, raw_prompt: list[dict[str, str]]) -> list[int]:
        """Tokenize one raw prompt using the model tokenizer chat template.

        Args:
            raw_prompt: Chat-message prompt format from `RLHFDataset`.

        Returns:
            Flat prompt token id list.
        """

        tokenized_prompt = self.tokenizer.apply_chat_template(
            raw_prompt,
            add_generation_prompt=True,
            tokenize=True,
        )
        return normalize_token_ids(tokenized_output=tokenized_prompt)

    def _build_branch_records(
        self, *, prompt_group: PromptGroup, prompt_ids: list[int], tree
    ) -> list[LeafBatchRecord]:
        """Build branch-aware rollout records from one completed branch tree.

        Args:
            prompt_group: Repeated prompt group for this rollout.
            prompt_ids: Tokenized prompt ids reused by all leaves.
            tree: Completed branch tree returned by the executor.

        Returns:
            Branch-aware rollout records for all tree leaves.
        """

        branch_records: list[LeafBatchRecord] = []
        for leaf in tree.leaves:
            branch_index = resolve_leaf_branch_metadata(
                tree=tree,
                leaf=leaf,
                prompt_uid=prompt_group.prompt_uid,
                selector_mode=self.settings.selector_mode,
            )
            branch_records.append(
                LeafBatchRecord(
                    prompt_ids=prompt_ids,
                    response_ids=list(leaf.token_ids),
                    response_logprobs=[token.logprob for token in leaf.tokens] if leaf.tokens else None,
                    reward_scores=build_reward_scores(branch_index=branch_index),
                    branch_index=branch_index,
                )
            )
        return branch_records

    async def _generate_prompt_group(
        self,
        *,
        prompt_group: PromptGroup,
        artifact_store: ArtifactStore,
    ) -> PromptGroupGeneration:
        """Generate branched rollouts for one prompt group.

        Args:
            prompt_group: Repeated prompt group to expand.
            artifact_store: Batch-scoped artifact store shared across prompt groups.

        Returns:
            Prompt-group records plus numeric branching metrics.
        """

        prompt_text = extract_prompt_text(raw_prompt=prompt_group.raw_prompt)
        prompt_ids = self._tokenize_prompt(raw_prompt=prompt_group.raw_prompt)
        doc_id = self.doc_counter
        self.doc_counter += 1
        executor = self._build_executor(
            prompt_text=prompt_text,
            doc_id=doc_id,
            artifact_store=artifact_store,
        )
        tree = await executor.run_branching_rollouts_async(
            doc_id=doc_id,
            doc_attempt=0,
            task_name="branching_dapo_train",
            model_id="branching_dapo",
        )
        branch_records = self._build_branch_records(
            prompt_group=prompt_group,
            prompt_ids=prompt_ids,
            tree=tree,
        )
        max_possible_leaf_count = self.settings.leaf_limit()
        assert len(branch_records) <= max_possible_leaf_count, (
            "Branch tree exceeded the configured branching leaf capacity."
        )
        realized_leaf_count = len(branch_records)
        group_metrics = {
            **summarize_rollout_records(records=branch_records),
            **executor.metrics(),
            "branching/max_possible_leaf_count": float(max_possible_leaf_count),
            "branching/realized_leaf_count": float(realized_leaf_count),
            "branching/unrealized_leaf_count": float(
                max(max_possible_leaf_count - realized_leaf_count, 0)
            ),
            "branching/realization_rate": float(
                realized_leaf_count / max_possible_leaf_count
                if max_possible_leaf_count
                else 0.0
            ),
            "branching/prompt_group_count": 1.0,
        }
        return PromptGroupGeneration(records=branch_records, metrics=group_metrics)

    def _pad_prompt_ids(self, prompt_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-pad one prompt token sequence to rollout prompt length.

        Args:
            prompt_ids: Unpadded prompt token ids.

        Returns:
            Tuple of padded prompt ids and prompt attention mask.
        """

        assert len(prompt_ids) <= self.rollout_config.prompt_length, "Prompt exceeds rollout prompt_length."
        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": prompt_ids},
            padding="max_length",
            max_length=self.rollout_config.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids_tensor = prompt_output["input_ids"]
        prompt_attention_mask = prompt_output["attention_mask"]
        if prompt_ids_tensor.dim() == 1:
            prompt_ids_tensor = prompt_ids_tensor.unsqueeze(0)
            prompt_attention_mask = prompt_attention_mask.unsqueeze(0)
        return prompt_ids_tensor, prompt_attention_mask

    def _pad_response_ids(
        self, response_ids: list[int], response_logprobs: list[float] | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Right-pad one response token sequence to rollout response length.

        Args:
            response_ids: Unpadded response token ids.
            response_logprobs: Optional per-token logprobs.

        Returns:
            Padded response ids, attention mask, response mask, and logprobs.
        """

        assert len(response_ids) <= self.rollout_config.response_length, "Response exceeds rollout response_length."
        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": response_ids},
            padding="max_length",
            max_length=self.rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids_tensor = response_output["input_ids"]
        response_attention_mask = response_output["attention_mask"]
        if response_ids_tensor.dim() == 1:
            response_ids_tensor = response_ids_tensor.unsqueeze(0)
            response_attention_mask = response_attention_mask.unsqueeze(0)
        response_mask = response_attention_mask.clone()
        logprobs_tensor = None
        if response_logprobs is not None:
            pad_size = self.rollout_config.response_length - len(response_logprobs)
            logprobs_tensor = torch.tensor(response_logprobs + [0.0] * pad_size).unsqueeze(0)
        return response_ids_tensor, response_attention_mask, response_mask, logprobs_tensor

    def _aggregate_group_metrics(self, generations: list[PromptGroupGeneration]) -> dict[str, float]:
        """Aggregate prompt-group metrics across a rollout batch.

        Args:
            generations: Prompt-group generation payloads.

        Returns:
            Batch-level numeric branching metrics.
        """

        if not generations:
            return {
                "branching/prompt_group_count": 0.0,
                "branching/leaf_count": 0.0,
                "branching/branch_point_count": 0.0,
                "branching/cluster_count_mean": 0.0,
                "branching/max_possible_leaf_count": 0.0,
                "branching/realized_leaf_count": 0.0,
                "branching/unrealized_leaf_count": 0.0,
                "branching/realization_rate": 0.0,
            }
        sum_keys = {
            "branching/prompt_group_count",
            "branching/leaf_count",
            "branching/branch_point_count",
            "branching/max_possible_leaf_count",
            "branching/realized_leaf_count",
            "branching/unrealized_leaf_count",
        }
        metric_keys = {key for generation in generations for key in generation.metrics}
        aggregated: dict[str, float] = {}
        for metric_key in metric_keys:
            values = [generation.metrics.get(metric_key, 0.0) for generation in generations]
            aggregated[metric_key] = float(sum(values)) if metric_key in sum_keys else float(np.mean(values))
        return aggregated

    def _build_non_tensor_batch(
        self,
        *,
        outputs: list[LeafBatchRecord],
        prompt_group_by_uid: dict[str, PromptGroup],
    ) -> dict[str, np.ndarray]:
        """Build non-tensor rollout payloads needed by reward computation.

        Args:
            outputs: Realized leaf rollout records for this batch.
            prompt_group_by_uid: Prompt-group lookup keyed by prompt uid.

        Returns:
            Non-tensor batch payload aligned with `outputs`.
        """

        reward_scores = []
        prompt_uid_values = []
        raw_prompts = []
        data_sources = []
        extra_infos = []
        reward_models = []
        for output in outputs:
            prompt_uid = output.branch_index.prompt_uid
            prompt_group = prompt_group_by_uid[prompt_uid]
            reward_scores.append(output.reward_scores)
            prompt_uid_values.append(prompt_uid)
            raw_prompts.append(prompt_group.raw_prompt)
            data_sources.append(prompt_group.data_source)
            extra_infos.append(prompt_group.extra_info)
            reward_models.append(prompt_group.reward_model)
        return {
            "__num_turns__": np.array([1] * len(outputs), dtype=np.int32),
            "uid": np.array(prompt_uid_values, dtype=object),
            "reward_scores": np.array(reward_scores, dtype=object),
            "raw_prompt": np.array(raw_prompts, dtype=object),
            "data_source": np.array(data_sources, dtype=object),
            "extra_info": np.array(extra_infos, dtype=object),
            "reward_model": np.array(reward_models, dtype=object),
        }

    async def _attach_reward_outputs(self, *, data: DataProto) -> DataProto:
        """Populate `rm_scores` and reward metadata using the async reward loop.

        Args:
            data: Rollout batch with prompt metadata and decoded responses.

        Returns:
            Updated rollout batch containing `rm_scores` and reward extra info.
        """

        if self.reward_loop_worker_handles is None:
            return data
        reward_results = await asyncio.gather(
            *[
                random.choice(self.reward_loop_worker_handles).compute_score.remote(data[row_index : row_index + 1])
                for row_index in range(len(data))
            ]
        )
        prompt_length = data.batch["prompts"].size(1)
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1
        rm_scores = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        rm_scores[torch.arange(rm_scores.size(0)), valid_response_length] = torch.tensor(
            [float(result["reward_score"]) for result in reward_results],
            dtype=torch.float32,
        )
        data.batch["rm_scores"] = rm_scores
        reward_extra_infos = [dict(result.get("reward_extra_info", {})) for result in reward_results]
        reward_extra_keys = list(reward_extra_infos[0].keys()) if reward_extra_infos else []
        for reward_extra_key in reward_extra_keys:
            data.non_tensor_batch[reward_extra_key] = np.array(
                [reward_info.get(reward_extra_key) for reward_info in reward_extra_infos],
                dtype=object,
            )
        data.meta_info["reward_extra_keys"] = reward_extra_keys
        return data

    def _build_dataproto(
        self,
        outputs: list[LeafBatchRecord],
        elapsed_seconds: float,
        batch_metrics: dict[str, float],
        prompt_group_by_uid: dict[str, PromptGroup],
    ) -> DataProto:
        """Pack branch rollout outputs into the DataProto structure expected by `verl`.

        Args:
            outputs: Unpadded leaf batch records.
            elapsed_seconds: Wall-clock generation time in seconds.
            prompt_group_by_uid: Prompt-group lookup used to restore reward inputs.

        Returns:
            DataProto batch ready for union with the trainer batch.
        """

        assert outputs, "Branching rollouts must realize at least one leaf."
        prompt_tensors = []
        response_tensors = []
        response_masks = []
        attention_masks = []
        input_ids = []
        position_ids = []
        rollout_log_probs = []
        for output in outputs:
            padded_prompt_ids, prompt_attention_mask = self._pad_prompt_ids(prompt_ids=output.prompt_ids)
            padded_response_ids, response_attention_mask, response_mask, response_logprobs = self._pad_response_ids(
                response_ids=output.response_ids,
                response_logprobs=output.response_logprobs,
            )
            attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
            sample_input_ids = torch.cat([padded_prompt_ids, padded_response_ids], dim=1)
            sample_position_ids = compute_position_id_with_mask(attention_mask)
            prompt_tensors.append(padded_prompt_ids)
            response_tensors.append(padded_response_ids)
            response_masks.append(response_mask)
            attention_masks.append(attention_mask)
            input_ids.append(sample_input_ids)
            position_ids.append(sample_position_ids)
            if response_logprobs is not None:
                rollout_log_probs.append(response_logprobs)

        batch = TensorDict(
            {
                "prompts": torch.cat(prompt_tensors, dim=0),
                "responses": torch.cat(response_tensors, dim=0),
                "response_mask": torch.cat(response_masks, dim=0),
                "attention_mask": torch.cat(attention_masks, dim=0),
                "input_ids": torch.cat(input_ids, dim=0),
                "position_ids": torch.cat(position_ids, dim=0),
            },
            batch_size=[len(outputs)],
        )
        if len(rollout_log_probs) == len(outputs):
            batch["rollout_log_probs"] = torch.cat(rollout_log_probs, dim=0)
        non_tensor_batch = self._build_non_tensor_batch(
            outputs=outputs,
            prompt_group_by_uid=prompt_group_by_uid,
        )
        timing_metrics = {
            "branching/generate_sequences/mean": elapsed_seconds,
            **batch_metrics,
        }
        record_generation_metrics(metrics=timing_metrics)
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"timing": timing_metrics},
        )

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate branched rollouts grouped by repeated prompt uid.

        Args:
            prompts: Repeated prompt batch passed by the PPO trainer.

        Returns:
            Rollout outputs with all realized leaf trajectories per prompt.
        """

        start_time = time.perf_counter()
        self.doc_counter = 0
        artifact_store = self._build_batch_artifact_store(prompts=prompts)
        prompt_groups = build_prompt_groups(
            non_tensor_batch=prompts.non_tensor_batch,
            expected_group_size=1,
        )
        generations = await asyncio.gather(
            *[
                self._generate_prompt_group(
                    prompt_group=prompt_group,
                    artifact_store=artifact_store,
                )
                for prompt_group in prompt_groups
            ]
        )
        outputs = [record for generation in generations for record in generation.records]
        batch_metrics = self._aggregate_group_metrics(generations=generations)
        elapsed_seconds = time.perf_counter() - start_time
        prompt_group_by_uid = {prompt_group.prompt_uid: prompt_group for prompt_group in prompt_groups}
        data = self._build_dataproto(
            outputs=outputs,
            elapsed_seconds=elapsed_seconds,
            batch_metrics=batch_metrics,
            prompt_group_by_uid=prompt_group_by_uid,
        )
        return await self._attach_reward_outputs(data=data)
