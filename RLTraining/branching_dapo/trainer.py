"""Repo-local PPO trainer wrapper that logs branching-specific runtime metrics."""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportArgumentType=false, reportUnusedCoroutine=false

from __future__ import annotations

import os
import uuid
from pprint import pprint
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from branching_dapo.runtime_metrics import consume_runtime_metrics
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


class BranchingRayPPOTrainer(RayPPOTrainer):
    """Repo-local PPO trainer that preserves upstream behavior and logs branching metrics."""

    @staticmethod
    def _hybrid_fsdp_dispatch_info(*, world_size: int, fsdp_size: int) -> tuple[list[int], list[bool]] | None:
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
        assert world_size % fsdp_size == 0, f"world_size={world_size} must be divisible by fsdp_size={fsdp_size}"
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
        dispatch_info = self._hybrid_fsdp_dispatch_info(world_size=worker_group.world_size, fsdp_size=fsdp_size)
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
        return os.environ.get("BRANCHING_FIT_DEBUG") == "1" or experiment_name.startswith("debug_fit_")

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
        padded_batch, pad_size = pad_dataproto_to_divisor(data=batch, size_divisor=actor_dp_size)
        if pad_size == 0:
            return padded_batch, pad_size
        metrics[f"trainer/{stage_name}_pad_size"] = pad_size
        metrics[f"trainer/{stage_name}_padded_batch_size"] = len(padded_batch)
        self._fit_debug(message=f"pad_{stage_name} batch_size={len(batch)} pad_size={pad_size} dp_size={actor_dp_size}")
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
        return bool(rollout_corr_config and rollout_corr_config.get("bypass_mode", False))

    def _should_balance_branching_batch(self, *, batch: DataProto, metrics: dict[str, float | int | object]) -> bool:
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
        self._fit_debug(message=f"skip_balance_batch batch_size={len(batch)} dp_size={dp_size}")
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
        for value in metric_values if isinstance(metric_values, (list, tuple)) else [metric_values]:
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
                flattened_values = self._flatten_numeric_metric_values(metric_values=metric_value)
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
            self._fit_debug(message=f"skip_worker_metrics names={','.join(skipped_metric_names)}")

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
                norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                config=self.config.algorithm,
            )

        original_uid = batch.non_tensor_batch["uid"]
        batch.non_tensor_batch["uid"] = batch.non_tensor_batch["branch_uid"]
        try:
            return compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                config=self.config.algorithm,
            )
        finally:
            batch.non_tensor_batch["uid"] = original_uid

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

        assert self.config.algorithm.adv_estimator != AdvantageEstimator.REMAX, (
            "REMAX is unsupported for ragged branching rollouts."
        )

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
            assert uid_string not in uid_index, f"Duplicate source uid encountered: {uid_string}"
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

        assert len(generation_batch) > 0, "Generation batch must contain at least one realized leaf."
        assert "uid" in generation_batch.non_tensor_batch, "Generation batch must preserve source uid values."
        uid_index = BranchingRayPPOTrainer._build_uid_index(
            uid_values=source_non_tensor_batch["uid"]
        )
        selected_indices = np.array(
            [uid_index[str(uid_value)] for uid_value in generation_batch.non_tensor_batch["uid"]],
            dtype=np.int64,
        )
        if source_batch.batch is not None and generation_batch.batch is not None:
            missing_batch_keys = [
                key for key in source_batch.batch.keys() if key not in generation_batch.batch.keys()
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

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
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
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                source_non_tensor_batch = self._snapshot_non_tensor_batch(batch=batch)

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.async_rollout_manager.start_profile()
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
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
                    if self._should_balance_branching_batch(batch=batch, metrics=metrics):
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    batch.meta_info["images_seqlens"] = self._collect_image_seqlens(batch=batch)
                    self._drop_shadowing_non_tensor_keys(batch=batch)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        self._fit_debug(message="reward_start")
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)
                        self._fit_debug(message="reward_complete")

                    self._mark_batch_auto_padding(batch=batch)

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = self._should_bypass_old_log_prob_recompute(
                        batch=batch,
                        metrics=metrics,
                        rollout_corr_config=rollout_corr_config,
                    )
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            self._fit_debug(message="old_log_prob_start")
                            padded_batch, old_log_prob_pad_size = self._pad_for_actor_dp(
                                batch=batch,
                                metrics=metrics,
                                stage_name="old_log_prob",
                            )
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(padded_batch)
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
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                router_mode = getattr(self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled")
                                if router_mode == "R2":
                                    batch.batch.pop("routed_experts")
                                else:
                                    old_log_prob.batch.pop("routed_experts")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            self._fit_debug(message="ref_log_prob_start")
                            padded_batch, ref_log_prob_pad_size = self._pad_for_actor_dp(
                                batch=batch,
                                metrics=metrics,
                                stage_name="ref_log_prob",
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
                            values = self._unpad_output_batch(batch=values, pad_size=values_pad_size)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        self._fit_debug(message="advantage_start")
                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({key: np.array(value) for key, value in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
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
                        self._reduce_worker_metrics(worker_metrics=critic_output.meta_info["metrics"], metrics=metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            padded_batch, actor_pad_size = self._pad_for_actor_dp(
                                batch=batch,
                                metrics=metrics,
                                stage_name="update_actor",
                            )
                            actor_output = self._update_actor(padded_batch)
                            if actor_pad_size:
                                metrics["trainer/update_actor_used_padded_batch"] = 1
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
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)
                        self._fit_debug(message="update_weights_complete")
                        self._reduce_worker_metrics(worker_metrics=actor_output.meta_info["metrics"], metrics=metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
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
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch,
                        timing_raw=timing_raw,
                        n_gpus=self.resource_pool_manager.get_n_gpus(),
                    )
                )
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                metrics.update(consume_runtime_metrics())

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
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}",
                        sub_dir=f"step{self.global_steps}",
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
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
