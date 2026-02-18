"""Build modular steer-branching report data and static viewer shell."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from candidate_clustering import ClusteringConfig, cluster_candidates_by_step
from io_utils import read_json, read_jsonl
from static_report import (
    ReportOutput,
    build_report_payload,
    render_report_html,
    report_bundle_payload,
)


def parse_args() -> argparse.Namespace:
    """Parse report bundle CLI arguments.

    Returns:
        Parsed argument namespace.
    """

    parser = argparse.ArgumentParser(
        description="Build static report viewer from one or more steer-branching runs."
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
        "--run-dir",
        action="append",
        type=Path,
        required=True,
        help="Run artifact directory. Repeat to include multiple outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output HTML path (default: <first run>/report.html).",
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
        help="Optional clustering cache path override."
        " For multi-run mode, each run gets <run-dir>/<cache-name>.",
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


def resolve_cluster_cache_path(*, run_dir: Path, cluster_cache: Path | None) -> Path:
    """Resolve per-run clustering cache path.

    Args:
        run_dir: Source run directory.
        cluster_cache: Optional user override.

    Returns:
        Absolute cache path for this run.
    """

    if cluster_cache is None:
        return (run_dir / "cluster_prompt_cache.json").resolve()
    if cluster_cache.name:
        return (run_dir / cluster_cache.name).resolve()
    return (run_dir / "cluster_prompt_cache.json").resolve()


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
        cache_path=resolve_cluster_cache_path(
            run_dir=run_dir, cluster_cache=cluster_cache
        ),
        env_paths=resolve_env_paths(run_dir=run_dir, cli_env_paths=env_files),
    )


def slug_text(*, value: str) -> str:
    """Build a filesystem-safe id segment.

    Args:
        value: Source text.

    Returns:
        Lowercase slug with alphanumeric and dash chars.
    """

    lowered = value.lower()
    chars = [char if char.isalnum() else "-" for char in lowered]
    slug = "".join(chars)
    compact = "-".join(part for part in slug.split("-") if part)
    return compact[:64] or "output"


def output_id(*, run_dir: Path, config: dict[str, Any], index: int) -> str:
    """Build deterministic output id.

    Args:
        run_dir: Source run directory.
        config: Run config mapping.
        index: Zero-based bundle index.

    Returns:
        Stable output id string.
    """

    prompt = str(config.get("prompt", "")).strip()
    base = slug_text(value=prompt or run_dir.name)
    return f"{base}-{index + 1}"


def build_report_payload_for_run(
    *,
    run_dir: Path,
    disable_clustering: bool,
    gemini_model: str,
    gemini_temperature: float,
    previous_steps_window: int,
    cluster_max_concurrency: int,
    cluster_cache: Path | None,
    cluster_seed_override: int | None,
    env_files: list[Path],
) -> dict[str, Any]:
    """Build report payload for one run directory.

    Args:
        run_dir: Source run directory.
        disable_clustering: Clustering toggle.
        gemini_model: Gemini model id.
        gemini_temperature: Clustering prompt temperature.
        previous_steps_window: Selected-step prompt context size.
        cluster_max_concurrency: Max concurrent clustering requests.
        cluster_cache: Optional cache path override.
        cluster_seed_override: Optional seed override.
        env_files: Optional dotenv file paths.

    Returns:
        JSON-serializable report payload for one run.
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
        env_files=env_files,
    )
    clustering = cluster_candidates_by_step(
        candidates=candidates,
        config=cluster_config,
        steps=steps,
    )
    return build_report_payload(
        config=config,
        steps=steps,
        candidates=candidates,
        token_stats=token_stats,
        final_text=final_text,
        clustering=clustering,
    )


def resolve_run_dirs(*, run_dirs: list[Path]) -> list[Path]:
    """Resolve and deduplicate run directories while preserving order.

    Args:
        run_dirs: CLI-provided run dirs.

    Returns:
        Ordered list of unique resolved run dirs.
    """

    unique: list[Path] = []
    seen: set[Path] = set()
    for run_dir in run_dirs:
        resolved = run_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def build_bundle_outputs(
    *,
    run_dirs: list[Path],
    disable_clustering: bool,
    gemini_model: str,
    gemini_temperature: float,
    previous_steps_window: int,
    cluster_max_concurrency: int,
    cluster_cache: Path | None,
    cluster_seed_override: int | None,
    env_files: list[Path],
) -> list[ReportOutput]:
    """Build bundled outputs for multiple run directories.

    Args:
        run_dirs: Source run directories.
        disable_clustering: Clustering toggle.
        gemini_model: Gemini model id.
        gemini_temperature: Clustering prompt temperature.
        previous_steps_window: Selected-step prompt context size.
        cluster_max_concurrency: Max concurrent clustering requests.
        cluster_cache: Optional cache path override.
        cluster_seed_override: Optional seed override.
        env_files: Optional dotenv file paths.

    Returns:
        Ordered report output entries.
    """

    outputs: list[ReportOutput] = []
    for index, run_dir in enumerate(run_dirs):
        payload = build_report_payload_for_run(
            run_dir=run_dir,
            disable_clustering=disable_clustering,
            gemini_model=gemini_model,
            gemini_temperature=gemini_temperature,
            previous_steps_window=previous_steps_window,
            cluster_max_concurrency=cluster_max_concurrency,
            cluster_cache=cluster_cache,
            cluster_seed_override=cluster_seed_override,
            env_files=env_files,
        )
        raw_config = payload.get("config")
        config: dict[str, Any] = raw_config if isinstance(raw_config, dict) else {}
        output_prompt = str(config.get("prompt", "")).strip()
        outputs.append(
            ReportOutput(
                output_id=output_id(run_dir=run_dir, config=config, index=index),
                prompt=output_prompt,
                run_dir=str(run_dir),
                report_payload=payload,
            )
        )
    return outputs


def report_asset_source_dir() -> Path:
    """Return source directory containing viewer assets.

    Returns:
        Absolute asset source directory path.
    """

    return (
        Path(__file__).resolve().parent / "report_viewer" / "report_assets"
    ).resolve()


def copy_report_assets(*, output_path: Path) -> None:
    """Copy report CSS/JS assets next to output HTML.

    Args:
        output_path: Written report HTML path.

    Returns:
        None.

    Example:
        >>> copy_report_assets(output_path=Path("output/report.html"))  # doctest: +SKIP
    """

    source_dir = report_asset_source_dir()
    destination_dir = output_path.parent / "report_assets"
    destination_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("styles.css", "app.js"):
        shutil.copy2(source_dir / filename, destination_dir / filename)


def build_report(
    *,
    run_dirs: list[Path],
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
    """Build a bundled report viewer from one or more run artifacts.

    Args:
        run_dirs: Run artifact directories.
        output_path: Optional output HTML path.
        disable_clustering: Disables Gemini clustering when true.
        gemini_model: Gemini model id for clustering.
        gemini_temperature: Clustering prompt temperature.
        previous_steps_window: Previous selected-step context window.
        cluster_max_concurrency: Max concurrent Gemini requests.
        cluster_cache: Optional clustering cache filename override.
        cluster_seed_override: Optional seed override.
        env_files: Additional dotenv paths.

    Returns:
        Written report HTML path.

    Example:
        >>> build_report(run_dirs=[Path("output/my_run")], output_path=None)  # doctest: +SKIP
        PosixPath('output/my_run/report.html')
    """

    resolved_run_dirs = resolve_run_dirs(run_dirs=run_dirs)
    assert resolved_run_dirs, "Expected at least one run directory."
    outputs = build_bundle_outputs(
        run_dirs=resolved_run_dirs,
        disable_clustering=disable_clustering,
        gemini_model=gemini_model,
        gemini_temperature=gemini_temperature,
        previous_steps_window=previous_steps_window,
        cluster_max_concurrency=cluster_max_concurrency,
        cluster_cache=cluster_cache,
        cluster_seed_override=cluster_seed_override,
        env_files=env_files or [],
    )
    bundle = report_bundle_payload(outputs=outputs)
    report_html = render_report_html(report_bundle=bundle)
    resolved_output = output_path or (resolved_run_dirs[0] / "report.html")
    resolved_output.write_text(report_html, encoding="utf-8")
    copy_report_assets(output_path=resolved_output)
    return resolved_output


def main() -> None:
    """Run report rebuild CLI."""

    args = parse_args()
    output_path = build_report(
        run_dirs=list(args.run_dir),
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
