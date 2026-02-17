#!/usr/bin/env python3
"""Filter LoRA checkpoints so adapters are loadable by vLLM.

This script removes non-LoRA tensors from `adapter_model.safetensors` and writes
cleaned checkpoints to sibling `-vllm` directories by default.

Example:
    python cleanup_lora_checkpoints.py \
      --run-dir /fs/scratch/.../SFTTraining/outputs/qwen3_8b_to_think_aimek4_lora_b1_ga12 \
      --apply \
      --overwrite
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from re import Pattern, compile

from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

ADAPTER_FILE_NAME = "adapter_model.safetensors"
DEFAULT_OUTPUT_SUFFIX = "-vllm"
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
CHECKPOINT_NAME_PATTERN: Pattern[str] = compile(r"^checkpoint-(\d+)$")
LORA_MARKERS = (
    "lora_A",
    "lora_B",
    "lora_embedding_A",
    "lora_embedding_B",
    "lora_magnitude_vector",
)


@dataclass(frozen=True)
class CheckpointCleanupPlan:
    """Cleanup plan for one checkpoint directory.

    Inputs:
        source_dir: Original checkpoint directory.
        output_dir: Destination directory for cleaned adapter output.
        source_adapter_path: Original adapter safetensors path.
        output_adapter_path: Destination adapter safetensors path.
        kept_tensor_keys: Tensor keys to keep in output adapter.
        dropped_tensor_keys: Tensor keys to remove from output adapter.
        metadata: Metadata copied from source safetensors file.

    Outputs:
        Immutable plan used by dry-run and apply phases.

    Example:
        >>> plan = CheckpointCleanupPlan(
        ...     source_dir=Path("checkpoint-100"),
        ...     output_dir=Path("checkpoint-100-vllm"),
        ...     source_adapter_path=Path("checkpoint-100/adapter_model.safetensors"),
        ...     output_adapter_path=Path("checkpoint-100-vllm/adapter_model.safetensors"),
        ...     kept_tensor_keys=("x",),
        ...     dropped_tensor_keys=("y",),
        ...     metadata={},
        ... )
        >>> plan.total_tensors()
        2
    """

    source_dir: Path
    output_dir: Path
    source_adapter_path: Path
    output_adapter_path: Path
    kept_tensor_keys: tuple[str, ...]
    dropped_tensor_keys: tuple[str, ...]
    metadata: dict[str, str]

    def total_tensors(self) -> int:
        """Return total tensor count before filtering.

        Inputs:
            None.

        Outputs:
            Total tensor key count in the source adapter.
        """

        return len(self.kept_tensor_keys) + len(self.dropped_tensor_keys)


def bytes_to_gib(num_bytes: int) -> float:
    """Convert byte count to GiB.

    Inputs:
        num_bytes: Raw size in bytes.

    Outputs:
        Size in gibibytes.
    """

    return float(num_bytes) / (1024.0**3)


def add_source_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add run/checkpoint source selection arguments to parser.

    Inputs:
        parser: Argument parser instance to mutate.

    Outputs:
        None. Adds source-selection CLI arguments.
    """

    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        type=Path,
        help="Run output directory containing checkpoint-<step> subdirectories.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        default=[],
        type=Path,
        help="Explicit checkpoint directory to clean. Repeatable.",
    )


def add_tensor_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add LoRA tensor filtering arguments to parser.

    Inputs:
        parser: Argument parser instance to mutate.

    Outputs:
        None. Adds tensor-filtering CLI arguments.
    """

    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=list(DEFAULT_TARGET_MODULES),
        help="Target module names to keep (LoRA tensors only).",
    )


def add_write_mode_args(parser: argparse.ArgumentParser) -> None:
    """Add output and execution mode arguments to parser.

    Inputs:
        parser: Argument parser instance to mutate.

    Outputs:
        None. Adds output and apply-mode CLI arguments.
    """

    parser.add_argument(
        "--output-suffix",
        default=DEFAULT_OUTPUT_SUFFIX,
        help="Suffix added to cleaned checkpoint directories.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite adapter tensors inside the original checkpoint directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output directories.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write cleaned adapters. Without this flag, run as dry-run.",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for LoRA checkpoint cleanup.

    Inputs:
        None.

    Outputs:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    add_source_selection_args(parser=parser)
    add_tensor_filter_args(parser=parser)
    add_write_mode_args(parser=parser)
    return parser


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for checkpoint tensor cleanup.

    Inputs:
        None (reads from command line).

    Outputs:
        Parsed argument namespace.
    """

    parser = build_arg_parser()
    args = parser.parse_args()
    assert args.run_dir or args.checkpoint_dir, "Specify --run-dir or --checkpoint-dir."
    assert args.in_place or args.output_suffix, "--output-suffix must not be empty."
    return args


def checkpoint_step_from_name(checkpoint_name: str) -> int | None:
    """Parse `checkpoint-<step>` directory names.

    Inputs:
        checkpoint_name: Directory basename.

    Outputs:
        Parsed step for matching names, otherwise `None`.
    """

    name_match = CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint_name)
    if name_match is None:
        return None
    return int(name_match.group(1))


def discover_checkpoint_dirs(run_dir: Path) -> list[Path]:
    """Discover train checkpoint directories in one run folder.

    Inputs:
        run_dir: Training run output directory.

    Outputs:
        Sorted checkpoint directory paths matching `checkpoint-<step>`.
    """

    assert run_dir.exists(), f"Run directory does not exist: {run_dir}"
    assert run_dir.is_dir(), f"Run directory is not a directory: {run_dir}"
    step_dirs: list[tuple[int, Path]] = []
    for child_path in run_dir.iterdir():
        if not child_path.is_dir():
            continue
        step_value = checkpoint_step_from_name(checkpoint_name=child_path.name)
        if step_value is None:
            continue
        step_dirs.append((step_value, child_path))
    step_dirs.sort(key=lambda item: item[0])
    return [path for _, path in step_dirs]


def dedupe_paths(paths: list[Path]) -> list[Path]:
    """Resolve and deduplicate paths while preserving order.

    Inputs:
        paths: Candidate paths to normalize and deduplicate.

    Outputs:
        Stable list of unique resolved paths.
    """

    unique_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for path in paths:
        resolved_path = path.expanduser().resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        unique_paths.append(resolved_path)
    return unique_paths


def collect_source_checkpoint_dirs(*, run_dirs: list[Path], checkpoint_dirs: list[Path]) -> list[Path]:
    """Collect source checkpoints from run-level and explicit inputs.

    Inputs:
        run_dirs: Run directories scanned for `checkpoint-<step>`.
        checkpoint_dirs: Explicit checkpoint directories.

    Outputs:
        Dedupe-resolved checkpoint directory paths.
    """

    discovered_dirs: list[Path] = []
    for run_dir in run_dirs:
        discovered_dirs.extend(discover_checkpoint_dirs(run_dir=run_dir))
    discovered_dirs.extend(checkpoint_dirs)
    resolved_dirs = dedupe_paths(paths=discovered_dirs)
    for checkpoint_dir in resolved_dirs:
        assert checkpoint_dir.exists(), f"Checkpoint directory does not exist: {checkpoint_dir}"
        assert checkpoint_dir.is_dir(), f"Checkpoint path is not a directory: {checkpoint_dir}"
    return resolved_dirs


def resolve_output_dir(*, checkpoint_dir: Path, output_suffix: str, in_place: bool) -> Path:
    """Resolve output directory for cleaned checkpoint artifacts.

    Inputs:
        checkpoint_dir: Source checkpoint directory.
        output_suffix: Suffix for sibling output directory naming.
        in_place: Whether to rewrite source directory in-place.

    Outputs:
        Destination directory path.
    """

    if in_place:
        return checkpoint_dir
    return checkpoint_dir.with_name(f"{checkpoint_dir.name}{output_suffix}")


def is_vllm_lora_tensor(*, tensor_key: str, target_modules: set[str]) -> bool:
    """Return `True` for LoRA tensor keys that should remain for vLLM.

    Inputs:
        tensor_key: Safetensors parameter key.
        target_modules: Allowed module names (e.g., `q_proj`, `gate_proj`).

    Outputs:
        Boolean keep/drop decision for this key.

    Example:
        >>> is_vllm_lora_tensor(
        ...     tensor_key='...self_attn.q_proj.lora_A.weight',
        ...     target_modules={'q_proj'},
        ... )
        True
    """

    has_lora_marker = any(marker in tensor_key for marker in LORA_MARKERS)
    if not has_lora_marker:
        return False
    key_tokens = set(tensor_key.split("."))
    return any(module_name in key_tokens for module_name in target_modules)


def split_tensor_keys(*, adapter_path: Path, target_modules: set[str]) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]:
    """Split adapter tensor keys into kept and dropped groups.

    Inputs:
        adapter_path: Source `adapter_model.safetensors` file.
        target_modules: Allowed module names for kept LoRA tensors.

    Outputs:
        `(kept_keys, dropped_keys, metadata)` tuple.
    """

    kept_keys: list[str] = []
    dropped_keys: list[str] = []
    with safe_open(filename=str(adapter_path), framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        for tensor_key in sorted(reader.keys()):
            if is_vllm_lora_tensor(tensor_key=tensor_key, target_modules=target_modules):
                kept_keys.append(tensor_key)
            else:
                dropped_keys.append(tensor_key)
    return tuple(kept_keys), tuple(dropped_keys), metadata


def build_cleanup_plan(*, checkpoint_dir: Path, output_suffix: str, in_place: bool, target_modules: set[str]) -> CheckpointCleanupPlan:
    """Build a cleanup plan for one checkpoint directory.

    Inputs:
        checkpoint_dir: Source checkpoint directory.
        output_suffix: Destination suffix when not running in-place.
        in_place: Whether to write output back to source directory.
        target_modules: Allowed module names for LoRA tensor retention.

    Outputs:
        Planned cleanup data for this checkpoint.
    """

    source_adapter_path = checkpoint_dir / ADAPTER_FILE_NAME
    assert source_adapter_path.exists(), f"Missing adapter file: {source_adapter_path}"
    output_dir = resolve_output_dir(
        checkpoint_dir=checkpoint_dir,
        output_suffix=output_suffix,
        in_place=in_place,
    )
    output_adapter_path = output_dir / ADAPTER_FILE_NAME
    kept_keys, dropped_keys, metadata = split_tensor_keys(
        adapter_path=source_adapter_path,
        target_modules=target_modules,
    )
    assert kept_keys, f"No vLLM LoRA tensors retained for {source_adapter_path}"
    return CheckpointCleanupPlan(
        source_dir=checkpoint_dir,
        output_dir=output_dir,
        source_adapter_path=source_adapter_path,
        output_adapter_path=output_adapter_path,
        kept_tensor_keys=kept_keys,
        dropped_tensor_keys=dropped_keys,
        metadata=metadata,
    )


def prepare_output_dir(*, output_dir: Path, overwrite: bool, in_place: bool) -> None:
    """Create destination directory and enforce overwrite policy.

    Inputs:
        output_dir: Target output directory.
        overwrite: Whether existing output directories may be replaced.
        in_place: Whether source directory is modified directly.

    Outputs:
        None. Creates directories and may delete stale output directories.
    """

    if in_place:
        return
    if output_dir.exists():
        assert overwrite, f"Output directory exists, pass --overwrite: {output_dir}"
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_non_adapter_files(*, source_dir: Path, output_dir: Path, in_place: bool) -> None:
    """Copy non-adapter artifacts to output checkpoint directory.

    Inputs:
        source_dir: Source checkpoint directory.
        output_dir: Destination checkpoint directory.
        in_place: Skip copy when writing in-place.

    Outputs:
        None. Copies files and subdirectories except adapter safetensors.
    """

    if in_place:
        return
    for child_path in source_dir.iterdir():
        if child_path.name == ADAPTER_FILE_NAME:
            continue
        destination_path = output_dir / child_path.name
        if child_path.is_dir():
            shutil.copytree(src=child_path, dst=destination_path)
            continue
        shutil.copy2(src=child_path, dst=destination_path)


def write_filtered_adapter(*, plan: CheckpointCleanupPlan, target_modules: set[str]) -> int:
    """Write cleaned adapter safetensors for one checkpoint.

    Inputs:
        plan: Per-checkpoint cleanup plan.
        target_modules: Allowed module names for LoRA tensor retention.

    Outputs:
        Output adapter size in bytes.
    """

    kept_tensors: dict[str, Tensor] = {}
    with safe_open(filename=str(plan.source_adapter_path), framework="pt", device="cpu") as reader:
        for tensor_key in reader.keys():
            if not is_vllm_lora_tensor(tensor_key=tensor_key, target_modules=target_modules):
                continue
            kept_tensors[tensor_key] = reader.get_tensor(tensor_key)
    tmp_output_path = plan.output_adapter_path.with_suffix(".tmp.safetensors")
    save_file(
        tensors=kept_tensors,
        filename=str(tmp_output_path),
        metadata=plan.metadata or None,
    )
    tmp_output_path.replace(plan.output_adapter_path)
    return plan.output_adapter_path.stat().st_size


def print_plan(plan: CheckpointCleanupPlan) -> None:
    """Print a concise dry-run summary for one checkpoint.

    Inputs:
        plan: Planned cleanup data for one checkpoint.

    Outputs:
        None. Writes a summary to stdout.
    """

    source_size = plan.source_adapter_path.stat().st_size
    drop_count = len(plan.dropped_tensor_keys)
    keep_count = len(plan.kept_tensor_keys)
    print(f"\nCheckpoint: {plan.source_dir}")
    print(f"Output: {plan.output_dir}")
    print(f"Tensor keys: keep={keep_count}, drop={drop_count}, total={plan.total_tensors()}")
    print(f"Source adapter size: ~{bytes_to_gib(num_bytes=source_size):.2f} GiB")
    if not plan.dropped_tensor_keys:
        return
    preview = ", ".join(plan.dropped_tensor_keys[:4])
    print(f"Dropped key preview: {preview}")


def main() -> int:
    """Run LoRA checkpoint tensor cleanup for vLLM.

    Inputs:
        CLI arguments from `sys.argv`.

    Outputs:
        Process exit code (`0` on success).
    """

    args = parse_args()
    target_modules = set(args.target_modules)
    assert target_modules, "--target-modules must not be empty."
    source_dirs = collect_source_checkpoint_dirs(
        run_dirs=args.run_dir,
        checkpoint_dirs=args.checkpoint_dir,
    )
    plans = [
        build_cleanup_plan(
            checkpoint_dir=source_dir,
            output_suffix=args.output_suffix,
            in_place=args.in_place,
            target_modules=target_modules,
        )
        for source_dir in source_dirs
    ]
    for plan in plans:
        print_plan(plan=plan)
    if not args.apply:
        print("\nDry run only. Re-run with --apply to write cleaned adapters.")
        return 0
    for plan in plans:
        prepare_output_dir(
            output_dir=plan.output_dir,
            overwrite=args.overwrite,
            in_place=args.in_place,
        )
        copy_non_adapter_files(
            source_dir=plan.source_dir,
            output_dir=plan.output_dir,
            in_place=args.in_place,
        )
        output_bytes = write_filtered_adapter(
            plan=plan,
            target_modules=target_modules,
        )
        print(f"Wrote: {plan.output_adapter_path} (~{bytes_to_gib(num_bytes=output_bytes):.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
