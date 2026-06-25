"""Repo-local PPO trainer wrapper that logs branching-specific runtime metrics."""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportArgumentType=false, reportUnusedCoroutine=false

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from branching_dapo.advantage import (
    BranchSegmentAdvantage,
    compute_branch_segment_advantages,
)
from branching_dapo.config_types import BranchingRolloutSettings
from branching_dapo.runtime_metrics import consume_runtime_metrics
from branching_dapo.update_masking import (
    build_steer_only_response_mask,
    exclude_response_token_spans,
)
from verl import DataProto
from verl.protocol import DataProtoConfig, pad_dataproto_to_divisor, unpad_dataproto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip

REWARD_COMPONENT_KEYS = ("structure_reward", "answer_reward")
BLOCK_METRIC_KEYS = (
    ("exec", "exec_block_count", "exec_block_word_count"),
    ("steer", "steer_block_count", "steer_block_word_count"),
)


@dataclass(frozen=True)
class ActorUpdateMaskState:
    """Original actor-update tensors restored after a masked update."""

    response_mask: torch.Tensor
    advantages: torch.Tensor | None


REPEAT_BLOCK_KINDS = ("exec", "steer")


def _finite_numeric_values(*, values: object) -> np.ndarray:
    """Return finite float values from reward-extra arrays."""

    numeric_values: list[float] = []
    for raw_value in np.asarray(values, dtype=object).reshape(-1):
        if raw_value is None:
            continue
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value):
            numeric_values.append(numeric_value)
    return np.asarray(numeric_values, dtype=np.float64)


def _finite_numeric_values_by_index(*, values: object) -> list[float | None]:
    """Return finite float values while preserving reward-extra row positions."""

    numeric_values: list[float | None] = []
    for raw_value in np.asarray(values, dtype=object).reshape(-1):
        if raw_value is None:
            numeric_values.append(None)
            continue
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            numeric_values.append(None)
            continue
        if np.isfinite(numeric_value):
            numeric_values.append(numeric_value)
        else:
            numeric_values.append(None)
    return numeric_values


def _string_values_by_index(*, values: object) -> list[str | None]:
    """Return non-empty string values while preserving reward-extra positions."""

    string_values: list[str | None] = []
    for raw_value in np.asarray(values, dtype=object).reshape(-1):
        if raw_value is None:
            string_values.append(None)
            continue
        normalized_text = " ".join(str(raw_value).strip().split())
        if normalized_text and normalized_text.lower() not in {"none", "nan"}:
            string_values.append(normalized_text)
        else:
            string_values.append(None)
    return string_values


def _normalized_answer_text(*, value: object) -> str | None:
    """Return a whitespace-normalized answer string for diversity metrics."""

    if value is None:
        return None
    normalized_text = " ".join(str(value).strip().split())
    if not normalized_text:
        return None
    return normalized_text


def _prompt_uid_from_branch_uid(*, value: object) -> str | None:
    """Extract prompt uid from serialized branch metadata."""

    if value is None:
        return None
    branch_payload = json.loads(str(value))
    prompt_uid = branch_payload.get("prompt_uid")
    if prompt_uid is None:
        return None
    prompt_uid_text = str(prompt_uid).strip()
    if not prompt_uid_text:
        return None
    return prompt_uid_text


class BranchingRayPPOTrainer(RayPPOTrainer):
    """Repo-local PPO trainer that preserves upstream behavior and logs branching metrics."""

    @staticmethod
    def _hybrid_fsdp_dispatch_info(
        *, world_size: int, fsdp_size: int
    ) -> tuple[list[int], list[bool]] | None:
        """Return corrected dispatch/collect info for hybrid-sharded FSDP workers.

        Args:
            world_size: Total number of worker ranks in the worker group.
            fsdp_size: FSDP shard size configured for the worker group.

        Returns:
            Tuple of dispatch ranks and collect mask when the worker group is in a
            hybrid-shard layout (`1 < fsdp_size < world_size`). Returns `None`
            when no correction is needed.
        """

        if fsdp_size <= 1 or fsdp_size >= world_size:
            return None
        assert (
            world_size % fsdp_size == 0
        ), f"world_size={world_size} must be divisible by fsdp_size={fsdp_size}"
        dispatch_ranks = [rank // fsdp_size for rank in range(world_size)]
        collect_mask = [rank % fsdp_size == 0 for rank in range(world_size)]
        return dispatch_ranks, collect_mask

    def _override_worker_group_dispatch_info(
        self,
        *,
        worker_group: object | None,
        mesh_name: str,
        fsdp_size: int,
    ) -> None:
        """Override lazy dispatch metadata for hybrid-sharded worker groups.

        Args:
            worker_group: Worker group whose cached lazy dispatch metadata should be corrected.
            mesh_name: Dispatch mesh name to override.
            fsdp_size: FSDP shard size configured for the worker group role.

        Returns:
            None.
        """

        if worker_group is None:
            return
        dispatch_info = self._hybrid_fsdp_dispatch_info(
            world_size=worker_group.world_size, fsdp_size=fsdp_size
        )
        if dispatch_info is None:
            return
        dispatch_ranks, collect_mask = dispatch_info
        worker_group._dispatch_info[mesh_name] = dispatch_ranks
        worker_group._collect_info[mesh_name] = collect_mask
        self._fit_debug(
            message=(
                f"override_dispatch mesh={mesh_name} world_size={worker_group.world_size} "
                f"fsdp_size={fsdp_size} dispatch={dispatch_ranks} collect={collect_mask}"
            )
        )

    def init_workers(self) -> None:
        """Initialize workers and correct hybrid-FSDP lazy dispatch metadata."""

        super().init_workers()
        self._override_worker_group_dispatch_info(
            worker_group=self.actor_rollout_wg,
            mesh_name="actor",
            fsdp_size=int(self.config.actor_rollout_ref.actor.fsdp_config.fsdp_size),
        )
        if self.use_reference_policy:
            self._override_worker_group_dispatch_info(
                worker_group=self.ref_policy_wg,
                mesh_name="actor",
                fsdp_size=int(self.config.actor_rollout_ref.ref.fsdp_config.fsdp_size),
            )
        if self.use_critic:
            self._override_worker_group_dispatch_info(
                worker_group=self.critic_wg,
                mesh_name="critic",
                fsdp_size=int(self.config.critic.fsdp_config.fsdp_size),
            )

    def _fit_debug_enabled(self) -> bool:
        """Return whether fit-loop debug markers should be emitted.

        Args:
            None.

        Returns:
            True when fit debugging is enabled through the environment or the
            experiment name.
        """

        experiment_name = str(self.config.trainer.experiment_name)
        return os.environ.get(
            "BRANCHING_FIT_DEBUG"
        ) == "1" or experiment_name.startswith("debug_fit_")

    def _fit_debug(self, *, message: str) -> None:
        """Emit a flushed fit-loop debug marker when debug tracing is enabled.

        Args:
            message: Human-readable marker describing the current control-flow point.

        Returns:
            None.
        """

        if not self._fit_debug_enabled():
            return
        step_label = getattr(self, "global_steps", "init")
        print(f"[fit-debug step={step_label}] {message}", flush=True)

    def _actor_dp_size(self) -> int:
        """Return the actor data-parallel size used by distributed compute steps.

        Args:
            None.

        Returns:
            Actor-side data-parallel size.
        """

        return int(self._get_dp_size(self.actor_rollout_wg, "actor"))

    def _mark_batch_auto_padding(self, *, batch: DataProto) -> None:
        """Enable `DataProto` auto-padding for ragged branching batches.

        Args:
            batch: Training batch produced from realized branching leaves.

        Returns:
            None.
        """

        batch.meta_info[DataProtoConfig.auto_padding_key] = True

    def _stage_padding_divisor(self, *, stage_name: str) -> int:
        """Return the global batch divisor required before distributed dispatch."""

        actor_dp_size = self._actor_dp_size()
        if stage_name == "update_actor":
            mini_batch_size = self._actor_update_mini_batch_size()
            return math.lcm(actor_dp_size, mini_batch_size)
        if stage_name == "update_critic":
            mini_batch_size = self._critic_update_mini_batch_size()
            return math.lcm(actor_dp_size, mini_batch_size)
        return actor_dp_size

    def _actor_update_mini_batch_size(self) -> int:
        """Return the global PPO mini-batch size used by verl's actor update."""

        actor_config = self.config.actor_rollout_ref.actor
        rollout_config = self.config.actor_rollout_ref.rollout
        mini_batch_size = int(actor_config.ppo_mini_batch_size) * int(rollout_config.n)
        assert mini_batch_size > 0
        return mini_batch_size

    def _critic_update_mini_batch_size(self) -> int:
        """Return the global PPO mini-batch size used by verl's critic update."""

        critic_config = self.config.critic
        rollout_config = self.config.actor_rollout_ref.rollout
        mini_batch_size = int(critic_config.ppo_mini_batch_size) * int(rollout_config.n)
        assert mini_batch_size > 0
        return mini_batch_size

    def _pad_for_actor_dp(
        self,
        *,
        batch: DataProto,
        metrics: dict[str, float | int | object],
        stage_name: str,
    ) -> tuple[DataProto, int]:
        """Pad a batch so actor-side lazy dispatch can split it across DP ranks.

        Args:
            batch: Realized training batch before a distributed actor-side compute.
            metrics: Metrics dictionary updated with padding metadata.
            stage_name: Human-readable stage label for debugging/metrics.

        Returns:
            Tuple of padded batch and pad size.
        """

        actor_dp_size = self._actor_dp_size()
        padding_divisor = self._stage_padding_divisor(stage_name=stage_name)
        padded_batch, pad_size = pad_dataproto_to_divisor(
            data=batch, size_divisor=padding_divisor
        )
        if pad_size == 0:
            return padded_batch, pad_size
        metrics[f"trainer/{stage_name}_pad_size"] = pad_size
        metrics[f"trainer/{stage_name}_padded_batch_size"] = len(padded_batch)
        metrics[f"trainer/{stage_name}_pad_divisor"] = padding_divisor
        self._fit_debug(
            message=(
                f"pad_{stage_name} batch_size={len(batch)} pad_size={pad_size} "
                f"dp_size={actor_dp_size} divisor={padding_divisor}"
            )
        )
        return padded_batch, pad_size

    @staticmethod
    def _unpad_output_batch(*, batch: DataProto, pad_size: int) -> DataProto:
        """Strip synthetic padding rows from a distributed worker output batch.

        Args:
            batch: Output batch produced from padded distributed compute.
            pad_size: Number of synthetic rows appended before dispatch.

        Returns:
            Output batch restored to the original unpadded batch size.
        """

        return unpad_dataproto(data=batch, pad_size=pad_size)

    def _should_bypass_old_log_prob_recompute(
        self,
        *,
        batch: DataProto,
        metrics: dict[str, float | int | object],
        rollout_corr_config: object | None,
    ) -> bool:
        """Return whether rollout log-probs should replace old-logprob recompute.

        Args:
            batch: Training batch produced from realized branching leaves.
            metrics: Metrics dictionary updated in-place with bypass metadata.
            rollout_corr_config: Rollout-correction config from the trainer config.

        Returns:
            `True` when the trainer should use `rollout_log_probs` directly.
        """

        del batch
        del metrics
        return bool(
            rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        )

    def _should_balance_branching_batch(
        self, *, batch: DataProto, metrics: dict[str, float | int | object]
    ) -> bool:
        """Return whether sequence-length balancing is valid for the realized batch.

        Args:
            batch: Training batch produced from realized branching leaves.
            metrics: Metrics dictionary updated in-place with skip metadata.

        Returns:
            `True` when balancing is enabled and the batch divides evenly across actor DP ranks.
        """

        if not self.config.trainer.balance_batch:
            return False
        dp_size = self._actor_dp_size()
        if len(batch) % dp_size == 0:
            return True
        metrics["trainer/balance_batch_skipped_ragged"] = 1
        metrics["trainer/ragged_batch_size"] = len(batch)
        metrics["trainer/ragged_batch_dp_size"] = dp_size
        self._fit_debug(
            message=f"skip_balance_batch batch_size={len(batch)} dp_size={dp_size}"
        )
        return False

    @staticmethod
    def _flatten_numeric_metric_values(*, metric_values: object) -> list[float]:
        """Flatten one rank-reduced metric payload into scalar numeric values.

        Args:
            metric_values: Raw metric payload returned by distributed workers.

        Returns:
            Flat list of numeric values. Returns an empty list when coercion fails.
        """

        flattened_values: list[float] = []
        for value in (
            metric_values
            if isinstance(metric_values, (list, tuple))
            else [metric_values]
        ):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            array_value = np.asarray(value)
            if array_value.dtype == object:
                return []
            flattened_values.extend(array_value.astype(np.float64).reshape(-1).tolist())
        return flattened_values

    def _reduce_worker_metrics(
        self,
        *,
        worker_metrics: dict[str, Any],
        metrics: dict[str, float | int | object],
    ) -> None:
        """Reduce worker metrics while tolerating vector-valued debug payloads.

        Args:
            worker_metrics: Raw metrics emitted by distributed worker calls.
            metrics: Trainer metrics dictionary updated in-place.

        Returns:
            None.
        """

        skipped_metric_names: list[str] = []
        for metric_name, metric_value in worker_metrics.items():
            try:
                reduced_value = reduce_metrics({metric_name: metric_value})[metric_name]
            except (TypeError, ValueError):
                flattened_values = self._flatten_numeric_metric_values(
                    metric_values=metric_value
                )
                if not flattened_values:
                    skipped_metric_names.append(metric_name)
                    continue
                if "max" in metric_name:
                    reduced_value = float(np.max(flattened_values))
                elif "min" in metric_name:
                    reduced_value = float(np.min(flattened_values))
                else:
                    reduced_value = float(np.mean(flattened_values))
            metrics[metric_name] = reduced_value
        if skipped_metric_names:
            metrics["trainer/skipped_worker_metric_count"] = len(skipped_metric_names)
            self._fit_debug(
                message=f"skip_worker_metrics names={','.join(skipped_metric_names)}"
            )

    def _compute_advantages(self, *, batch: DataProto) -> DataProto:
        """Compute advantages while preserving the trainer's rollout-group uid.

        Args:
            batch: Training batch with rollout outputs and reward tensors.

        Returns:
            Batch updated with `advantages` and `returns`.
        """

        if (
            self.config.algorithm.adv_estimator != "branch_interpolated_grpo"
            or "branch_uid" not in batch.non_tensor_batch
        ):
            return compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=self.config.algorithm.get(
                    "norm_adv_by_std_in_grpo", True
                ),
                config=self.config.algorithm,
            )

        original_uid = batch.non_tensor_batch["uid"]
        batch.non_tensor_batch["uid"] = batch.non_tensor_batch["branch_uid"]
        try:
            updated_batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=self.config.algorithm.get(
                    "norm_adv_by_std_in_grpo", True
                ),
                config=self.config.algorithm,
            )
        finally:
            batch.non_tensor_batch["uid"] = original_uid
        updated_batch.non_tensor_batch["uid"] = original_uid
        self._persist_branch_segment_advantages(batch=updated_batch)
        return updated_batch

    def _persist_branch_segment_advantages(self, *, batch: DataProto) -> None:
        """Persist final combined token advantage summaries for the tree viewer."""

        if "advantages" not in batch.batch.keys():
            return
        if "branch_uid" not in batch.non_tensor_batch:
            return
        prompt_contexts = self._advantage_prompt_contexts(batch=batch)
        if not prompt_contexts:
            return
        segments = compute_branch_segment_advantages(
            advantages=batch.batch["advantages"],
            response_mask=self._policy_response_mask(batch=batch),
            index=batch.non_tensor_batch["branch_uid"],
        )
        if not segments:
            return
        from branching_eval.event_db import EventDatabase

        rows_by_db_path: dict[Path, list[dict[str, object]]] = {}
        for segment in segments:
            context_row = prompt_contexts.get(segment.prompt_uid)
            if context_row is None:
                continue
            db_path, event_context = context_row
            rows_by_db_path.setdefault(db_path, []).append(
                self._node_advantage_row(
                    segment=segment,
                    event_context=event_context,
                )
            )
        for db_path, rows in rows_by_db_path.items():
            db = EventDatabase(path=db_path)
            updated_at_event_index = db.last_event_index()
            db.upsert_node_advantage_rows(
                rows=[
                    {**row, "updated_at_event_index": updated_at_event_index}
                    for row in rows
                ]
            )

    def _prune_nonpersistent_branching_logs(
        self, *, batch: DataProto, global_step: int
    ) -> dict[str, int]:
        """Remove completed per-step tree logs that are outside the retention interval."""

        settings = BranchingRolloutSettings.from_config(config=self.config)
        metrics = {
            "branching/artifacts/persistent_log_interval_steps": (
                settings.persistent_log_interval_steps
            )
        }
        if settings.should_persist_step_logs(global_step=global_step):
            metrics["branching/artifacts/persisted_step_logs"] = 1
            metrics["branching/artifacts/pruned_step_log_dirs"] = 0
            return metrics

        pruned_count = 0
        for batch_dir in self._tree_event_batch_dirs(batch=batch):
            if not batch_dir.exists():
                continue
            shutil.rmtree(batch_dir)
            pruned_count += 1
        metrics["branching/artifacts/persisted_step_logs"] = 0
        metrics["branching/artifacts/pruned_step_log_dirs"] = pruned_count
        return metrics

    @staticmethod
    def _tree_event_batch_dirs(*, batch: DataProto) -> list[Path]:
        """Return unique batch directories that contain rollout tree event DBs."""

        db_paths = batch.non_tensor_batch.get("tree_events_db_path")
        if db_paths is None:
            return []
        batch_dirs: dict[str, Path] = {}
        for raw_db_path in np.asarray(db_paths, dtype=object).reshape(-1):
            db_path = Path(str(raw_db_path))
            assert (
                db_path.name == "tree_events.sqlite"
            ), f"Expected tree_events.sqlite path, got {db_path}"
            batch_dirs[str(db_path.parent)] = db_path.parent
        return sorted(batch_dirs.values(), key=str)

    def _advantage_prompt_contexts(
        self, *, batch: DataProto
    ) -> dict[str, tuple[Path, dict[str, object]]]:
        """Return prompt uid to artifact DB and event-context mapping."""

        db_paths = batch.non_tensor_batch.get("tree_events_db_path")
        reward_scores = batch.non_tensor_batch.get("reward_scores")
        if db_paths is None or reward_scores is None:
            return {}
        assert len(db_paths) == len(reward_scores), (
            "tree_events_db_path and reward_scores must align: "
            f"{len(db_paths)} != {len(reward_scores)}"
        )
        contexts: dict[str, tuple[Path, dict[str, object]]] = {}
        for raw_db_path, raw_scores in zip(db_paths, reward_scores):
            scores = dict(raw_scores)
            metadata = scores.get("branch_metadata")
            event_context = scores.get("event_context")
            if not isinstance(metadata, dict) or not isinstance(event_context, dict):
                continue
            prompt_uid = str(metadata.get("prompt_uid") or "")
            if prompt_uid:
                contexts[prompt_uid] = (Path(str(raw_db_path)), event_context)
        return contexts

    def _node_advantage_row(
        self,
        *,
        segment: BranchSegmentAdvantage,
        event_context: dict[str, object],
    ) -> dict[str, object]:
        """Return one typed SQLite row for a segment advantage."""

        return {
            "doc_id": int(event_context["doc_id"]),
            "doc_attempt": int(event_context["doc_attempt"]),
            "task_name": str(event_context["task_name"]),
            "model_id": str(event_context["model_id"]),
            "selector_mode": str(event_context["selector_mode"]),
            "prompt_uid": segment.prompt_uid,
            "branch_tree_id": segment.branch_tree_id,
            "parent_node_id": segment.parent_node_id,
            "child_node_id": segment.child_node_id,
            "branch_depth": segment.branch_depth,
            "token_start": segment.token_start,
            "token_end": segment.token_end,
            "mean_combined_advantage": segment.mean_combined_advantage,
            "token_count": segment.token_count,
            "leaf_count": segment.leaf_count,
        }

    def _apply_actor_update_mask(
        self, *, batch: DataProto, metrics: dict[str, float | int | object]
    ) -> ActorUpdateMaskState | None:
        """Temporarily narrow `response_mask` for the actor update."""

        update_mode = BranchingRolloutSettings.from_config(
            config=self.config
        ).validated_update_mode()
        metrics["actor/update_mode_steer_only"] = (
            1.0 if update_mode == "steer_only" else 0.0
        )
        full_response_mask = batch.batch["response_mask"]
        policy_response_mask, excluded_stats = exclude_response_token_spans(
            response_mask=full_response_mask,
            span_rows=self._off_policy_token_span_rows(batch=batch),
        )
        metrics.update(excluded_stats.as_metrics(prefix="actor/off_policy_mask"))
        full_advantages = (
            batch.batch["advantages"] if "advantages" in batch.batch.keys() else None
        )
        if update_mode == "all":
            if excluded_stats.excluded_token_count <= 0:
                return None
            batch.batch["response_mask"] = policy_response_mask
            return ActorUpdateMaskState(
                response_mask=full_response_mask, advantages=full_advantages
            )
        steer_mask, stats = build_steer_only_response_mask(
            responses=batch.batch["responses"],
            response_mask=policy_response_mask,
            tokenizer=self.tokenizer,
            steer_phase_token_spans=self._steer_phase_token_span_rows(batch=batch),
        )
        metrics.update(stats.as_metrics(prefix="actor/update_mask"))
        selected_token_ratio = (
            stats.selected_token_count / stats.response_token_count
            if stats.response_token_count
            else 0.0
        )
        gradient_scale = math.sqrt(selected_token_ratio)
        metrics["actor/update_mask/loss_scale"] = gradient_scale
        metrics["actor/update_mask/gradient_scale"] = gradient_scale
        batch.batch["response_mask"] = steer_mask
        if full_advantages is not None:
            batch.batch["advantages"] = full_advantages * gradient_scale
        return ActorUpdateMaskState(
            response_mask=full_response_mask, advantages=full_advantages
        )

    def _policy_response_mask(self, *, batch: DataProto) -> torch.Tensor:
        """Return response mask with rollout-injected spans excluded."""

        policy_response_mask, _ = exclude_response_token_spans(
            response_mask=batch.batch["response_mask"],
            span_rows=self._off_policy_token_span_rows(batch=batch),
        )
        return policy_response_mask

    @staticmethod
    def _steer_phase_token_span_rows(*, batch: DataProto) -> list[object] | None:
        """Return per-row steer-phase token spans from rollout reward metadata."""

        reward_scores = batch.non_tensor_batch.get("reward_scores")
        if reward_scores is None:
            return None
        span_rows: list[object] = []
        for reward_score in np.asarray(reward_scores, dtype=object).reshape(-1):
            assert isinstance(
                reward_score, dict
            ), "reward_scores rows must be dictionaries"
            span_rows.append(reward_score.get("steer_phase_token_spans", ()))
        return span_rows

    @staticmethod
    def _off_policy_token_span_rows(*, batch: DataProto) -> list[object] | None:
        """Return per-row rollout-injected token spans from reward metadata."""

        reward_scores = batch.non_tensor_batch.get("reward_scores")
        if reward_scores is None:
            return None
        span_rows: list[object] = []
        for reward_score in np.asarray(reward_scores, dtype=object).reshape(-1):
            assert isinstance(
                reward_score, dict
            ), "reward_scores rows must be dictionaries"
            span_rows.append(reward_score.get("off_policy_token_spans", ()))
        return span_rows

    def _collect_image_seqlens(self, *, batch: DataProto) -> list[int]:
        """Collect image sequence lengths from an optional multimodal batch payload.

        Args:
            batch: Trainer batch whose non-tensor payload may include multimodal data.

        Returns:
            Flattened image sequence lengths for metrics logging. Returns an empty
            list when the batch is text-only.
        """

        if "multi_modal_inputs" not in batch.non_tensor_batch:
            return []
        image_sequence_lengths: list[int] = []
        for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
            if not multi_modal_input or "image_grid_thw" not in multi_modal_input:
                continue
            image_sequence_lengths.extend(multi_modal_input["images_seqlens"].tolist())
        return image_sequence_lengths

    def _assert_supported_adv_estimator(self) -> None:
        """Reject rollout estimators that still require rectangular generation.

        Args:
            None.

        Returns:
            None.
        """

        assert (
            self.config.algorithm.adv_estimator != AdvantageEstimator.REMAX
        ), "REMAX is unsupported for ragged branching rollouts."

    @staticmethod
    def _snapshot_non_tensor_batch(*, batch: DataProto) -> dict[str, np.ndarray]:
        """Copy source non-tensor arrays before generation mutates the batch.

        Args:
            batch: Source prompt batch before `_get_gen_batch`.

        Returns:
            Shallow copy of the source non-tensor batch arrays.
        """

        return {key: value.copy() for key, value in batch.non_tensor_batch.items()}

    @staticmethod
    def _repeat_generation_batch(
        *, generation_batch: DataProto, repeat_times: int
    ) -> DataProto:
        """Repeat prompts for grouped rollout generation using VERL semantics.

        Args:
            generation_batch: Prompt-only batch passed to the rollout manager.
            repeat_times: Number of response samples requested per source prompt.

        Returns:
            Interleaved repeated generation batch.
        """

        assert repeat_times >= 1, f"rollout.n must be positive, got {repeat_times}"
        if repeat_times == 1:
            return generation_batch
        return generation_batch.repeat(repeat_times=repeat_times, interleave=True)

    @staticmethod
    def _build_uid_index(*, uid_values: np.ndarray) -> dict[str, int]:
        """Build a stable lookup from source uid to source prompt row index.

        Args:
            uid_values: One uid per source prompt row.

        Returns:
            Mapping from uid string to source row index.
        """

        uid_index: dict[str, int] = {}
        for item_index, uid_value in enumerate(uid_values):
            uid_string = str(uid_value)
            assert (
                uid_string not in uid_index
            ), f"Duplicate source uid encountered: {uid_string}"
            uid_index[uid_string] = item_index
        return uid_index

    @staticmethod
    def _restore_generation_batch(
        *,
        source_batch: DataProto,
        source_non_tensor_batch: dict[str, np.ndarray],
        generation_batch: DataProto,
    ) -> DataProto:
        """Reattach source-side metadata to a ragged generation batch by uid.

        Args:
            source_batch: Original prompt batch with source tensor fields.
            source_non_tensor_batch: Snapshot of original prompt non-tensor fields.
            generation_batch: Ragged rollout outputs produced by the branching manager.

        Returns:
            Training batch aligned to realized rollout leaves.
        """

        assert (
            len(generation_batch) > 0
        ), "Generation batch must contain at least one realized leaf."
        assert (
            "uid" in generation_batch.non_tensor_batch
        ), "Generation batch must preserve source uid values."
        uid_index = BranchingRayPPOTrainer._build_uid_index(
            uid_values=source_non_tensor_batch["uid"]
        )
        selected_indices = np.array(
            [
                uid_index[str(uid_value)]
                for uid_value in generation_batch.non_tensor_batch["uid"]
            ],
            dtype=np.int64,
        )
        if source_batch.batch is not None and generation_batch.batch is not None:
            missing_batch_keys = [
                key
                for key in source_batch.batch.keys()
                if key not in generation_batch.batch.keys()
            ]
            if missing_batch_keys:
                aligned_source = source_batch.select_idxs(selected_indices).select(
                    batch_keys=missing_batch_keys,
                    non_tensor_batch_keys=[],
                )
                generation_batch = generation_batch.union(aligned_source)
        for key, values in source_non_tensor_batch.items():
            if key in generation_batch.non_tensor_batch:
                continue
            generation_batch.non_tensor_batch[key] = values[selected_indices]
        merged_meta_info = dict(source_batch.meta_info)
        merged_meta_info.update(generation_batch.meta_info)
        generation_batch.meta_info = merged_meta_info
        return generation_batch

    @staticmethod
    def _drop_shadowing_non_tensor_keys(*, batch: DataProto) -> None:
        """Remove non-tensor keys that duplicate tensor keys in place.

        Args:
            batch: Batch to sanitize before DataProto-to-TensorDict conversion.

        Returns:
            None.
        """

        shadow_keys = set(batch.batch.keys()) & set(batch.non_tensor_batch.keys())
        for key in shadow_keys:
            batch.non_tensor_batch.pop(key, None)

    @staticmethod
    def _reward_component_metrics(
        *, reward_extra_infos_dict: dict[str, object]
    ) -> dict[str, float]:
        """Reduce reward component arrays into scalar logger metrics."""

        metrics: dict[str, float] = {}
        for component_key in REWARD_COMPONENT_KEYS:
            raw_values = reward_extra_infos_dict.get(component_key)
            if raw_values is None:
                continue
            numeric_values = _finite_numeric_values(values=raw_values)
            if numeric_values.size == 0:
                continue
            metric_prefix = f"reward/{component_key}"
            metrics[f"{metric_prefix}/mean"] = float(np.mean(numeric_values))
            metrics[f"{metric_prefix}/max"] = float(np.max(numeric_values))
            metrics[f"{metric_prefix}/min"] = float(np.min(numeric_values))
        return metrics

    @staticmethod
    def _answer_diversity_metrics(
        *, reward_extra_infos_dict: dict[str, object]
    ) -> dict[str, float]:
        """Reduce answer diversity and prompt-level zero-answer-reward metrics."""

        metrics: dict[str, float] = {}
        raw_predictions = reward_extra_infos_dict.get("pred")
        if raw_predictions is not None:
            given_answers: list[str] = []
            for raw_prediction in np.asarray(raw_predictions, dtype=object).reshape(-1):
                normalized_answer = _normalized_answer_text(value=raw_prediction)
                if normalized_answer is not None:
                    given_answers.append(normalized_answer)
            answer_count = len(given_answers)
            unique_answer_count = len(set(given_answers))
            metrics["answer/given_answer_count"] = float(answer_count)
            metrics["answer/unique_given_answer_count"] = float(unique_answer_count)
            metrics["answer/unique_given_answer_ratio"] = (
                float(unique_answer_count / answer_count) if answer_count else 0.0
            )

        raw_branch_uids = reward_extra_infos_dict.get("branch_uid")
        raw_answer_rewards = reward_extra_infos_dict.get("answer_reward")
        if raw_branch_uids is None or raw_answer_rewards is None:
            return metrics

        prompt_rewards: dict[str, list[float]] = {}
        branch_uids = np.asarray(raw_branch_uids, dtype=object).reshape(-1)
        answer_rewards = _finite_numeric_values_by_index(values=raw_answer_rewards)
        for raw_branch_uid, answer_reward in zip(branch_uids.tolist(), answer_rewards):
            if answer_reward is None:
                continue
            prompt_uid = _prompt_uid_from_branch_uid(value=raw_branch_uid)
            if prompt_uid is None:
                continue
            prompt_rewards.setdefault(prompt_uid, []).append(answer_reward)
        if not prompt_rewards:
            return metrics

        prompt_count = len(prompt_rewards)
        prompt_pass_rates = [
            float(np.mean(rewards)) for rewards in prompt_rewards.values() if rewards
        ]
        zero_reward_prompt_count = sum(
            1
            for rewards in prompt_rewards.values()
            if rewards and all(reward == 0.0 for reward in rewards)
        )
        metrics["answer/prompt_count"] = float(prompt_count)
        metrics["answer/problem_pass_rate_mean"] = float(np.mean(prompt_pass_rates))
        metrics["answer/problem_pass_rate_max"] = float(np.max(prompt_pass_rates))
        metrics["answer/problem_pass_rate_min"] = float(np.min(prompt_pass_rates))
        metrics["answer/prompts_zero_answer_reward_all_rollouts_count"] = float(
            zero_reward_prompt_count
        )
        metrics["answer/prompts_zero_answer_reward_all_rollouts_ratio"] = float(
            zero_reward_prompt_count / prompt_count
        )
        return metrics

    @staticmethod
    def _block_structure_metrics(
        *, reward_extra_infos_dict: dict[str, object]
    ) -> dict[str, float]:
        """Reduce steer/exec block counts and lengths into scalar metrics."""

        metrics: dict[str, float] = {}
        for block_name, count_key, word_count_key in BLOCK_METRIC_KEYS:
            raw_counts = reward_extra_infos_dict.get(count_key)
            raw_word_counts = reward_extra_infos_dict.get(word_count_key)
            if raw_counts is None or raw_word_counts is None:
                continue
            block_counts = _finite_numeric_values(values=raw_counts)
            block_word_counts = _finite_numeric_values(values=raw_word_counts)
            if block_counts.size == 0 or block_word_counts.size == 0:
                continue
            total_blocks = float(np.sum(block_counts))
            total_words = float(np.sum(block_word_counts))
            metric_prefix = f"blocks/{block_name}"
            metrics[f"{metric_prefix}/num_blocks_total"] = total_blocks
            metrics[f"{metric_prefix}/num_blocks_per_leaf"] = float(
                np.mean(block_counts)
            )
            metrics[f"{metric_prefix}/words_total"] = total_words
            metrics[f"{metric_prefix}/words_per_leaf"] = float(
                np.mean(block_word_counts)
            )
            metrics[f"{metric_prefix}/avg_words_per_block"] = (
                total_words / total_blocks if total_blocks else 0.0
            )
        return metrics

    @staticmethod
    def _repetition_metrics(
        *, reward_extra_infos_dict: dict[str, object]
    ) -> dict[str, float]:
        """Reduce repeat-loop forced-close metadata into scalar metrics."""

        raw_forced = reward_extra_infos_dict.get("repeat_forced_think_close")
        if raw_forced is None:
            return {}
        forced_values = _finite_numeric_values_by_index(values=raw_forced)
        forced_flags = [value is not None and value > 0.5 for value in forced_values]
        leaf_count = len(forced_flags)
        if leaf_count == 0:
            return {}

        forced_count = sum(forced_flags)
        metrics: dict[str, float] = {
            "repetition/leaf_count": float(leaf_count),
            "repetition/forced_close_count": float(forced_count),
            "repetition/forced_close_ratio": float(forced_count / leaf_count),
        }
        kind_values = _string_values_by_index(
            values=reward_extra_infos_dict.get("repeat_block_kind", [])
        )
        count_values = _finite_numeric_values_by_index(
            values=reward_extra_infos_dict.get("repeat_block_count", [])
        )
        similarity_values = _finite_numeric_values_by_index(
            values=reward_extra_infos_dict.get("repeat_last_similarity_ratio", [])
        )

        for block_kind in REPEAT_BLOCK_KINDS:
            indices = [
                index
                for index, kind_value in enumerate(kind_values)
                if index < leaf_count
                and forced_flags[index]
                and kind_value == block_kind
            ]
            metric_prefix = f"repetition/{block_kind}"
            metrics[f"{metric_prefix}/forced_close_count"] = float(len(indices))
            metrics[f"{metric_prefix}/forced_close_ratio"] = float(
                len(indices) / leaf_count
            )
            repeated_counts: list[float] = []
            for index in indices:
                if index >= len(count_values):
                    continue
                count_value = count_values[index]
                if count_value is not None:
                    repeated_counts.append(count_value)
            similarities: list[float] = []
            for index in indices:
                if index >= len(similarity_values):
                    continue
                similarity_value = similarity_values[index]
                if similarity_value is not None:
                    similarities.append(similarity_value)
            if repeated_counts:
                metrics[f"{metric_prefix}/repeated_blocks_mean"] = float(
                    np.mean(repeated_counts)
                )
                metrics[f"{metric_prefix}/repeated_blocks_max"] = float(
                    np.max(repeated_counts)
                )
            if similarities:
                metrics[f"{metric_prefix}/last_similarity_mean"] = float(
                    np.mean(similarities)
                )
                metrics[f"{metric_prefix}/last_similarity_max"] = float(
                    np.max(similarities)
                )
        return metrics

    def fit(self) -> None:
        """Run PPO training while merging repo-local branching metrics into logger output."""

        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self._assert_supported_adv_estimator()

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        self._fit_debug(message="initial_update_weights_complete")
        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                self._fit_debug(message=f"batch_start epoch={epoch}")
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics: dict[str, float | int | object] = {}
                timing_raw: dict[str, float] = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = (
                    self.config.actor_rollout_ref.rollout.temperature
                )
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                source_non_tensor_batch = self._snapshot_non_tensor_batch(batch=batch)

                gen_batch = self._get_gen_batch(batch)
                gen_batch = self._repeat_generation_batch(
                    generation_batch=gen_batch,
                    repeat_times=int(self.config.actor_rollout_ref.rollout.n),
                )
                gen_batch.meta_info["global_steps"] = self.global_steps
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.async_rollout_manager.start_profile()
                        gen_batch_output = (
                            self.async_rollout_manager.generate_sequences(gen_batch)
                        )
                        self._fit_debug(message="generate_sequences_complete")
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.async_rollout_manager.stop_profile()
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    batch = self._restore_generation_batch(
                        source_batch=batch,
                        source_non_tensor_batch=source_non_tensor_batch,
                        generation_batch=gen_batch_output,
                    )

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if self._should_balance_branching_batch(
                        batch=batch, metrics=metrics
                    ):
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()
                    batch.meta_info["images_seqlens"] = self._collect_image_seqlens(
                        batch=batch
                    )
                    self._drop_shadowing_non_tensor_keys(batch=batch)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        self._fit_debug(message="reward_start")
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)
                        self._fit_debug(message="reward_complete")

                    self._mark_batch_auto_padding(batch=batch)

                    rollout_corr_config = self.config.algorithm.get(
                        "rollout_correction", None
                    )
                    bypass_recomputing_logprobs = (
                        self._should_bypass_old_log_prob_recompute(
                            batch=batch,
                            metrics=metrics,
                            rollout_corr_config=rollout_corr_config,
                        )
                    )
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import (
                            apply_bypass_mode,
                        )

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            self._fit_debug(message="old_log_prob_start")
                            padded_batch, old_log_prob_pad_size = (
                                self._pad_for_actor_dp(
                                    batch=batch,
                                    metrics=metrics,
                                    stage_name="old_log_prob",
                                )
                            )
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(
                                padded_batch
                            )
                            old_log_prob = self._unpad_output_batch(
                                batch=old_log_prob,
                                pad_size=old_log_prob_pad_size,
                            )
                            self._fit_debug(message="old_log_prob_complete")
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            metrics.update(
                                {
                                    "actor/entropy": entropy_agg.detach().item(),
                                    "perf/mfu/actor_infer": old_log_prob_mfu,
                                }
                            )
                            old_log_prob.batch.pop("entropys")
                            if (
                                "routed_experts" in batch.batch
                                and "routed_experts" in old_log_prob.batch
                            ):
                                router_mode = getattr(
                                    self.config.actor_rollout_ref.actor.router_replay,
                                    "mode",
                                    "disabled",
                                )
                                if router_mode == "R2":
                                    batch.batch.pop("routed_experts")
                                else:
                                    old_log_prob.batch.pop("routed_experts")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import (
                                    calculate_debug_metrics,
                                )

                                metrics.update(calculate_debug_metrics(batch))

                    assert (
                        "old_log_probs" in batch.batch
                    ), f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        with marked_timer(
                            str(Role.RefPolicy), timing_raw, color="olive"
                        ):
                            self._fit_debug(message="ref_log_prob_start")
                            padded_batch, ref_log_prob_pad_size = (
                                self._pad_for_actor_dp(
                                    batch=batch,
                                    metrics=metrics,
                                    stage_name="ref_log_prob",
                                )
                            )
                            ref_log_prob = self._compute_ref_log_prob(padded_batch)
                            ref_log_prob = self._unpad_output_batch(
                                batch=ref_log_prob,
                                pad_size=ref_log_prob_pad_size,
                            )
                            self._fit_debug(message="ref_log_prob_complete")
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            padded_batch, values_pad_size = self._pad_for_actor_dp(
                                batch=batch,
                                metrics=metrics,
                                stage_name="values",
                            )
                            values = self._compute_values(padded_batch)
                            values = self._unpad_output_batch(
                                batch=values, pad_size=values_pad_size
                            )
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        self._fit_debug(message="advantage_start")
                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {
                                    key: np.array(value)
                                    for key, value in reward_extra_infos_dict.items()
                                }
                            )

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch[
                                "token_level_scores"
                            ]

                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import (
                                compute_rollout_correction_and_add_to_batch,
                            )

                            batch, is_metrics = (
                                compute_rollout_correction_and_add_to_batch(
                                    batch, rollout_corr_config
                                )
                            )
                            metrics.update(is_metrics)

                        batch = self._compute_advantages(batch=batch)
                        self._fit_debug(message="advantages_complete")

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            padded_batch, critic_pad_size = self._pad_for_actor_dp(
                                batch=batch,
                                metrics=metrics,
                                stage_name="update_critic",
                            )
                            critic_output = self._update_critic(padded_batch)
                            if critic_pad_size:
                                metrics["trainer/update_critic_used_padded_batch"] = 1
                        self._reduce_worker_metrics(
                            worker_metrics=critic_output.meta_info["metrics"],
                            metrics=metrics,
                        )

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            update_mask_state = self._apply_actor_update_mask(
                                batch=batch, metrics=metrics
                            )
                            try:
                                padded_batch, actor_pad_size = self._pad_for_actor_dp(
                                    batch=batch,
                                    metrics=metrics,
                                    stage_name="update_actor",
                                )
                                actor_output = self._update_actor(padded_batch)
                                if actor_pad_size:
                                    metrics[
                                        "trainer/update_actor_used_padded_batch"
                                    ] = 1
                            finally:
                                if update_mask_state is not None:
                                    batch.batch["response_mask"] = (
                                        update_mask_state.response_mask
                                    )
                                    if update_mask_state.advantages is not None:
                                        batch.batch["advantages"] = (
                                            update_mask_state.advantages
                                        )
                        self._fit_debug(message="update_actor_complete")

                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print(
                                    "Force saving checkpoint: ESI instance expiration approaching."
                                )
                            with marked_timer(
                                "save_checkpoint", timing_raw, color="green"
                            ):
                                self._save_checkpoint()

                        if is_last_step:
                            metrics["trainer/skipped_final_update_weights"] = 1
                            self._fit_debug(message="skip_final_update_weights")
                        else:
                            with marked_timer(
                                "update_weights", timing_raw, color="red"
                            ):
                                self.checkpoint_manager.update_weights(
                                    self.global_steps
                                )
                            self._fit_debug(message="update_weights_complete")
                        self._reduce_worker_metrics(
                            worker_metrics=actor_output.meta_info["metrics"],
                            metrics=metrics,
                        )

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(
                            batch, reward_extra_infos_dict, timing_raw, rollout_data_dir
                        )

                if self.config.trainer.test_freq > 0 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                metrics.update(
                    {"training/global_step": self.global_steps, "training/epoch": epoch}
                )
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    self._reward_component_metrics(
                        reward_extra_infos_dict=reward_extra_infos_dict
                    )
                )
                metrics.update(
                    self._answer_diversity_metrics(
                        reward_extra_infos_dict=reward_extra_infos_dict
                    )
                )
                metrics.update(
                    self._block_structure_metrics(
                        reward_extra_infos_dict=reward_extra_infos_dict
                    )
                )
                metrics.update(
                    self._repetition_metrics(
                        reward_extra_infos_dict=reward_extra_infos_dict
                    )
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch,
                        timing_raw=timing_raw,
                        n_gpus=self.resource_pool_manager.get_n_gpus(),
                    )
                )
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(
                    compute_variance_proxy_metrics(
                        batch=batch, gradient_norm=gradient_norm
                    )
                )
                metrics.update(consume_runtime_metrics())
                metrics.update(
                    self._prune_nonpersistent_branching_logs(
                        batch=batch, global_step=self.global_steps
                    )
                )

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                self._fit_debug(message="logger_log_start")
                logger.log(data=metrics, step=self.global_steps)
                self._fit_debug(message="logger_log_complete")
                progress_bar.update(1)
                self.global_steps += 1
                self._fit_debug(message="progress_bar_advanced")

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool
                    == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}",
                        sub_dir=f"step{self.global_steps}",
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(
                            blocking=True
                        )
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
                    self._fit_debug(message="on_batch_end_complete")

        progress_bar.close()
        self._fit_debug(message="fit_loop_exhausted")
        raise AssertionError(
            "Training loop exhausted without hitting the normal final-step return. "
            f"global_steps={self.global_steps}, total_training_steps={self.total_training_steps}, "
            f"len(train_dataloader)={len(self.train_dataloader)}"
        )
