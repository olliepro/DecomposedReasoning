"""Rebuild interactive steer-branching report from artifact files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from candidate_clustering import ClusteringConfig, cluster_candidates_by_step
from io_utils import read_json, read_jsonl
from static_report import build_report_payload, render_report_html


def parse_args() -> argparse.Namespace:
    """Parse report rebuild CLI arguments.

    Returns:
        Parsed argument namespace.
    """

    parser = argparse.ArgumentParser(
        description="Rebuild static report from steer-branching artifacts."
    )
    add_io_args(parser=parser)
    add_cluster_args(parser=parser)
    return parser.parse_args()


def add_io_args(*, parser: argparse.ArgumentParser) -> None:
    """Add report I/O arguments.

    Args:
        parser: Target parser.

    Returns:
        None.
    """

    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Run artifact directory."
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional output HTML path."
    )


def add_cluster_args(*, parser: argparse.ArgumentParser) -> None:
    """Add prompt-clustering arguments.

    Args:
        parser: Target parser.

    Returns:
        None.
    """

    parser.add_argument(
        "--disable-clustering",
        action="store_true",
        help="Disable Gemini prompt clustering and use exact-text fallback.",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model id used for clustering.",
    )
    parser.add_argument(
        "--gemini-temperature",
        type=float,
        default=0.2,
        help="Gemini sampling temperature for clustering prompts.",
    )
    parser.add_argument(
        "--previous-steps-window",
        type=int,
        default=5,
        help="How many previous selected steps to include in cluster prompts.",
    )
    parser.add_argument(
        "--cluster-seed",
        type=int,
        default=None,
        help="Optional cluster seed override for deterministic fallback behavior.",
    )
    parser.add_argument(
        "--cluster-max-concurrency",
        type=int,
        default=50,
        help="Maximum concurrent Gemini clustering requests.",
    )
    parser.add_argument(
        "--cluster-cache",
        type=Path,
        default=None,
        help="Optional clustering cache path (default: <run-dir>/cluster_prompt_cache.json).",
    )
    parser.add_argument(
        "--env-file",
        action="append",
        type=Path,
        default=[],
        help="Dotenv file checked for Gemini key (repeatable).",
    )


def cluster_seed(*, config: dict[str, Any], override: int | None) -> int:
    """Resolve cluster seed from CLI override or run config.

    Args:
        config: Run config mapping.
        override: Optional explicit override.

    Returns:
        Seed value.
    """

    if override is not None:
        return int(override)
    value = config.get("seed", 0)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return 0


def project_root_dir() -> Path:
    """Resolve DecomposedReasoning repository root.

    Returns:
        Absolute repository root path.
    """

    return Path(__file__).resolve().parent.parent


def default_env_paths(*, run_dir: Path) -> tuple[Path, ...]:
    """Build default dotenv lookup order.

    Args:
        run_dir: Run directory.

    Returns:
        Ordered dotenv path tuple.
    """

    project_root = project_root_dir()
    return (
        project_root / ".env",
        project_root / "BuildSFTDataset/.env",
        Path(".env"),
        run_dir / ".env",
    )


def resolve_env_paths(*, run_dir: Path, cli_env_paths: list[Path]) -> tuple[Path, ...]:
    """Resolve ordered unique dotenv paths.

    Args:
        run_dir: Run directory.
        cli_env_paths: CLI-provided dotenv paths.

    Returns:
        Ordered unique path tuple.
    """

    ordered = [*default_env_paths(run_dir=run_dir), *cli_env_paths]
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in ordered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return tuple(unique)


def build_clustering_config(
    *,
    run_dir: Path,
    config: dict[str, Any],
    disable_clustering: bool,
    gemini_model: str,
    gemini_temperature: float,
    previous_steps_window: int,
    cluster_max_concurrency: int,
    cluster_cache: Path | None,
    cluster_seed_override: int | None,
    env_files: list[Path],
) -> ClusteringConfig:
    """Build prompt-clustering config for report generation.

    Args:
        run_dir: Run directory.
        config: Run config mapping.
        disable_clustering: Disable clustering toggle.
        gemini_model: Gemini model id.
        gemini_temperature: Clustering temperature.
        previous_steps_window: Previous-step context window.
        cluster_max_concurrency: Max concurrent Gemini requests.
        cluster_cache: Optional clustering cache path override.
        cluster_seed_override: Optional seed override.
        env_files: Additional dotenv paths.

    Returns:
        Prompt clustering configuration.
    """

    return ClusteringConfig(
        enabled=not disable_clustering,
        gemini_model=gemini_model,
        temperature=float(gemini_temperature),
        seed=cluster_seed(config=config, override=cluster_seed_override),
        previous_steps_window=max(1, int(previous_steps_window)),
        max_concurrent_requests=max(1, int(cluster_max_concurrency)),
        cache_path=(cluster_cache or (run_dir / "cluster_prompt_cache.json")).resolve(),
        env_paths=resolve_env_paths(run_dir=run_dir, cli_env_paths=env_files),
    )


def read_optional_jsonl(*, path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows from an optional artifact file.

    Args:
        path: Input JSONL path.

    Returns:
        Parsed row mappings, or an empty list when the file is missing.

    Example:
        >>> read_optional_jsonl(path=Path("missing.jsonl"))
        []
    """

    if not path.exists():
        return []
    return read_jsonl(path=path)


def build_report(
    *,
    run_dir: Path,
    output_path: Path | None,
    disable_clustering: bool = False,
    gemini_model: str = "gemini-3-flash-preview",
    gemini_temperature: float = 0.2,
    previous_steps_window: int = 5,
    cluster_max_concurrency: int = 50,
    cluster_cache: Path | None = None,
    cluster_seed_override: int | None = None,
    env_files: list[Path] | None = None,
) -> Path:
    """Build report HTML from run artifacts.

    Args:
        run_dir: Run artifact directory.
        output_path: Optional output path.
        disable_clustering: Disables Gemini clustering when true.
        gemini_model: Gemini model id for clustering.
        gemini_temperature: Clustering prompt temperature.
        previous_steps_window: Previous selected-step context window.
        cluster_max_concurrency: Max concurrent Gemini requests.
        cluster_cache: Optional clustering cache path override.
        cluster_seed_override: Optional seed override.
        env_files: Additional dotenv paths.

    Returns:
        Written report path.

    Example:
        >>> build_report(run_dir=Path("output/my_run"), output_path=None)  # doctest: +SKIP
        PosixPath('output/my_run/report.html')
    """

    config = read_json(path=run_dir / "config.json")
    steps = read_jsonl(path=run_dir / "steps.jsonl")
    candidates = read_optional_jsonl(path=run_dir / "steer_candidates.jsonl")
    token_stats = read_optional_jsonl(path=run_dir / "token_stats.jsonl")
    final_payload = read_json(path=run_dir / "final_text.json")
    final_text = str(final_payload.get("assistant_text", ""))
    cluster_config = build_clustering_config(
        run_dir=run_dir,
        config=config,
        disable_clustering=disable_clustering,
        gemini_model=gemini_model,
        gemini_temperature=gemini_temperature,
        previous_steps_window=previous_steps_window,
        cluster_max_concurrency=cluster_max_concurrency,
        cluster_cache=cluster_cache,
        cluster_seed_override=cluster_seed_override,
        env_files=env_files or [],
    )
    clustering = cluster_candidates_by_step(
        candidates=candidates,
        config=cluster_config,
        steps=steps,
    )
    report_payload = build_report_payload(
        config=config,
        steps=steps,
        candidates=candidates,
        token_stats=token_stats,
        final_text=final_text,
        clustering=clustering,
    )
    report_html = render_report_html(report_payload=report_payload)
    resolved_output = output_path or (run_dir / "report.html")
    resolved_output.write_text(report_html, encoding="utf-8")
    return resolved_output


def main() -> None:
    """Run report rebuild CLI."""

    args = parse_args()
    output_path = build_report(
        run_dir=args.run_dir,
        output_path=args.output,
        disable_clustering=bool(args.disable_clustering),
        gemini_model=str(args.gemini_model),
        gemini_temperature=float(args.gemini_temperature),
        previous_steps_window=int(args.previous_steps_window),
        cluster_max_concurrency=int(args.cluster_max_concurrency),
        cluster_cache=args.cluster_cache,
        cluster_seed_override=args.cluster_seed,
        env_files=list(args.env_file),
    )
    print(str(output_path))


if __name__ == "__main__":
    main()
