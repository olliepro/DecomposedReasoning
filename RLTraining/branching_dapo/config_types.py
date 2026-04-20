"""Typed config helpers for branching DAPO rollout and advantage settings."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VALID_SELECTOR_MODES = frozenset({"cluster_across", "random"})


@dataclass(frozen=True)
class BranchAdvantageIndex:
    """Serialized branch metadata used by the custom advantage estimator.

    Args:
        prompt_uid: Original repeated-prompt uid from `verl`.
        branch_tree_id: Stable branch-tree id for one prompt group.
        leaf_id: Stable leaf id inside the branch tree.
        leaf_node_id: Final tree node id for this leaf.
        path_node_ids: Ordered node ids from root to leaf.
        parent_branch_id: Parent node id for the final edge, or `None`.
        branch_depth: Number of branch edges on this leaf path.
        selected_cluster_id: Cluster id or label for the final branch choice.
        cluster_name: Human-readable cluster label when available.
        selector_mode: Selector mode used for this tree.
        candidate_pool_key: Candidate-pool cache key for the final branch point.

    Returns:
        Dataclass serialized into `uid` for the custom advantage estimator.
    """

    prompt_uid: str
    branch_tree_id: str
    leaf_id: str
    leaf_node_id: str
    path_node_ids: tuple[str, ...]
    parent_branch_id: str | None
    branch_depth: int
    selected_cluster_id: str | None
    cluster_name: str | None
    selector_mode: str
    candidate_pool_key: str | None

    def to_json(self) -> str:
        """Serialize branch metadata into a stable JSON string.

        Args:
            None.

        Returns:
            JSON string used as the batch `uid`.
        """

        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw_value: str) -> "BranchAdvantageIndex":
        """Parse a serialized branch-metadata string.

        Args:
            raw_value: JSON payload produced by `to_json`.

        Returns:
            Parsed `BranchAdvantageIndex`.
        """

        payload = json.loads(raw_value)
        assert isinstance(payload, dict), "Branch advantage index payload must be a mapping."
        path_node_ids = payload.get("path_node_ids", [])
        assert isinstance(path_node_ids, list), "path_node_ids must be a list."
        return cls(
            prompt_uid=str(payload["prompt_uid"]),
            branch_tree_id=str(payload["branch_tree_id"]),
            leaf_id=str(payload["leaf_id"]),
            leaf_node_id=str(payload["leaf_node_id"]),
            path_node_ids=tuple(str(node_id) for node_id in path_node_ids),
            parent_branch_id=None if payload.get("parent_branch_id") is None else str(payload["parent_branch_id"]),
            branch_depth=int(payload["branch_depth"]),
            selected_cluster_id=(
                None if payload.get("selected_cluster_id") is None else str(payload["selected_cluster_id"])
            ),
            cluster_name=None if payload.get("cluster_name") is None else str(payload["cluster_name"]),
            selector_mode=str(payload["selector_mode"]),
            candidate_pool_key=None if payload.get("candidate_pool_key") is None else str(payload["candidate_pool_key"]),
        )


@dataclass(frozen=True)
class BranchingRolloutSettings:
    """Repo-local rollout settings used by the branching manager.

    Args:
        selector_mode: Branch selector mode (`cluster_across` or `random`).
        branch_fanout: Number of selected candidates per branch point.
        max_branch_points_per_rollout: Maximum branch points on one path.
        num_candidates: Candidate count generated per branch point.
        max_clusters: Maximum clusters for `cluster_across`.
        branch_prob: Probability of branching at eligible steer boundaries.
        candidate_span_tokens: Span used by entropy-trigger candidate generation.
        max_steer_tokens: Maximum tokens generated for steer candidates.
        steer_repetition_penalty: Repetition penalty used for steer candidate generation.
        entropy_threshold: Optional explicit entropy-trigger threshold override.
        entropy_profile_name: Calibration profile label retained for parity with
            `Analysis/branching_eval` configs.
        max_concurrent_branches: Shared async branch-task limit.
        trigger_steer_enabled: Whether steer-boundary triggers are enabled.
        trigger_entropy_enabled: Whether entropy triggers are enabled.
        seed: Base RNG seed for prompt-group branching runs.
        cache_root: Root directory for branching artifacts.
        env_paths: Dotenv paths used for external selector API keys.

    Returns:
        Dataclass used by the rollout manager.
    """

    selector_mode: str = "cluster_across"
    branch_fanout: int = 4
    max_branch_points_per_rollout: int = 2
    num_candidates: int = 100
    max_clusters: int = 4
    branch_prob: float = 1.0
    candidate_span_tokens: int = 15
    max_steer_tokens: int = 15
    steer_repetition_penalty: float = 1.01
    entropy_threshold: float | None = None
    entropy_profile_name: str = "aime24_default"
    max_concurrent_branches: int = 64
    trigger_steer_enabled: bool = True
    trigger_entropy_enabled: bool = False
    seed: int = 0
    cache_root: Path = Path("/tmp/branching_dapo")
    env_paths: tuple[Path, ...] = ()

    @classmethod
    def from_config(cls, config: Any) -> "BranchingRolloutSettings":
        """Build rollout settings from the `rollout.custom.branching_dapo` mapping.

        Args:
            config: Top-level Hydra config or compatible mapping object.

        Returns:
            Parsed rollout settings with defaults filled in.
        """

        rollout_custom = cls._as_mapping(getattr(config.actor_rollout_ref.rollout, "custom", None))
        payload = cls._as_mapping(rollout_custom.get("branching_dapo"))
        default_cache_root = Path(os.environ.get("SCRATCH_ROOT", "/tmp")) / "branching_dapo"
        return cls(
            selector_mode=str(payload.get("selector_mode", "cluster_across")),
            branch_fanout=int(payload.get("branch_fanout", 4)),
            max_branch_points_per_rollout=int(payload.get("max_branch_points_per_rollout", 2)),
            num_candidates=int(payload.get("num_candidates", 100)),
            max_clusters=int(payload.get("max_clusters", 4)),
            branch_prob=float(payload.get("branch_prob", 1.0)),
            candidate_span_tokens=int(payload.get("candidate_span_tokens", 15)),
            max_steer_tokens=int(payload.get("max_steer_tokens", 15)),
            steer_repetition_penalty=float(payload.get("steer_repetition_penalty", 1.01)),
            entropy_threshold=(
                float(payload["entropy_threshold"])
                if payload.get("entropy_threshold") is not None
                else None
            ),
            entropy_profile_name=str(payload.get("entropy_profile_name", "aime24_default")),
            max_concurrent_branches=int(payload.get("max_concurrent_branches", 64)),
            trigger_steer_enabled=bool(payload.get("trigger_steer_enabled", True)),
            trigger_entropy_enabled=bool(payload.get("trigger_entropy_enabled", False)),
            seed=int(payload.get("seed", 0)),
            cache_root=Path(payload.get("cache_root", default_cache_root)),
            env_paths=cls._normalize_path_sequence(value=payload.get("env_paths")),
        )

    @staticmethod
    def _as_mapping(value: object) -> dict[str, Any]:
        """Normalize an OmegaConf-like value into a plain mapping.

        Args:
            value: Candidate mapping object.

        Returns:
            Plain string-keyed mapping.
        """

        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, Mapping):
            return {str(key): item for key, item in value.items()}
        return {}

    @staticmethod
    def _normalize_path_sequence(value: object) -> tuple[Path, ...]:
        """Normalize a config sequence of path-like values into `Path` objects.

        Args:
            value: Candidate sequence from Hydra or a plain Python config object.

        Returns:
            Tuple of normalized `Path` values, or an empty tuple when unset.

        Example:
            >>> BranchingRolloutSettings._normalize_path_sequence(value=["a.env", "b.env"])
            (Path('a.env'), Path('b.env'))
        """

        if value is None or isinstance(value, (str, bytes)):
            return ()
        if not isinstance(value, Sequence):
            return ()
        return tuple(Path(str(path_value)) for path_value in value)

    def leaf_limit(self) -> int:
        """Return the number of leaves produced by one branched rollout tree.

        Args:
            None.

        Returns:
            Branching leaf limit implied by fanout and depth.
        """

        return self.branch_fanout**self.max_branch_points_per_rollout

    def validated_selector_mode(self) -> str:
        """Validate and return the configured selector mode.

        Args:
            None.

        Returns:
            Supported selector-mode string.
        """

        assert self.selector_mode in VALID_SELECTOR_MODES, (
            f"Unsupported selector mode {self.selector_mode!r}. "
            f"Expected one of {sorted(VALID_SELECTOR_MODES)}."
        )
        return self.selector_mode

    @staticmethod
    def sanitize_path_component(raw_name: str) -> str:
        """Normalize a free-form label into a filesystem-safe path component.

        Args:
            raw_name: Candidate experiment, run, or batch label.

        Returns:
            Safe path component with punctuation collapsed to underscores.

        Example:
            >>> BranchingRolloutSettings.sanitize_path_component(raw_name="full scale/run")
            'full_scale_run'
        """

        collapsed_name = re.sub(pattern=r"[^A-Za-z0-9._-]+", repl="_", string=raw_name.strip())
        return collapsed_name.strip("._") or "branching_dapo"

    def artifact_root_dir(self) -> Path:
        """Return the root directory for all branching rollout artifacts.

        Args:
            None.

        Returns:
            Root artifact directory under `cache_root`.
        """

        return self.cache_root / "artifacts"

    def artifact_run_dir(self, *, run_name: str) -> Path:
        """Return the unique artifact directory for one training run.

        Args:
            run_name: Unique per-run label.

        Returns:
            Per-run artifact directory.
        """

        return self.artifact_root_dir() / self.sanitize_path_component(raw_name=run_name)

    def artifact_batch_dir(self, *, run_name: str, batch_name: str) -> Path:
        """Return the unique artifact directory for one rollout batch.

        Args:
            run_name: Unique per-run label.
            batch_name: Unique per-batch label within the run.

        Returns:
            Per-batch artifact directory.
        """

        return self.artifact_run_dir(run_name=run_name) / self.sanitize_path_component(raw_name=batch_name)

@dataclass(frozen=True)
class BranchAdvantageSettings:
    """Hyperparameters for recursive intra-branch advantage interpolation.

    Args:
        alpha: Weight on recursive intra-branch advantage.
        epsilon: Small constant for numerical stability.
        normalize_inter_by_std: Whether prompt-level inter advantage uses std normalization.
        normalize_intra_by_std: Whether local recursive branch deltas use std normalization.

    Returns:
        Dataclass consumed by the custom advantage estimator.
    """

    alpha: float = 0.5
    epsilon: float = 1e-6
    normalize_inter_by_std: bool = True
    normalize_intra_by_std: bool = True

    @classmethod
    def from_algorithm_config(cls, config: Any) -> "BranchAdvantageSettings":
        """Build advantage settings from the algorithm config mapping.

        Args:
            config: Algorithm config passed into the estimator.

        Returns:
            Parsed branch-advantage settings.
        """

        payload = BranchingRolloutSettings._as_mapping(config)
        return cls(
            alpha=float(payload.get("branching_alpha", 0.5)),
            epsilon=float(payload.get("branching_epsilon", 1e-6)),
            normalize_inter_by_std=bool(payload.get("norm_adv_by_std_in_grpo", True)),
            normalize_intra_by_std=bool(payload.get("branching_intra_norm_by_std", True)),
        )
