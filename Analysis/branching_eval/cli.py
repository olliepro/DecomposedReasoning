"""CLI entrypoint helpers for branching lm_eval experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from branching_eval.config_types import load_branching_eval_config
from branching_eval.run_matrix import run_experiment_matrix
from branching_eval.selector_types import SelectorMode


def parse_args() -> argparse.Namespace:
    """Parse branching-eval CLI arguments.

    Args:
        None.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(
        description="Run branching lm_eval experiments with tree artifacts."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--doc-id",
        type=int,
        action="append",
        default=None,
        help="Repeatable doc id override. Example: --doc-id 4",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        default=None,
        help="Existing run dir to resume from using tree_events.jsonl state.",
    )
    parser.add_argument(
        "--selector",
        type=str,
        choices=("cluster_across", "embed_diverse", "within_cluster", "random"),
        default=None,
    )
    return parser.parse_args()


def parse_doc_ids(*, raw_doc_ids: list[int] | None) -> tuple[int, ...] | None:
    """Return validated doc-id overrides from parsed CLI args.

    Args:
        raw_doc_ids: Optional repeated `--doc-id` values.

    Returns:
        Deduplicated ordered doc ids, or `None` when unset.

    Example:
        >>> parse_doc_ids(raw_doc_ids=[4, 4, 2])
        (4, 2)
    """

    if raw_doc_ids is None:
        return None
    deduped_doc_ids = tuple(dict.fromkeys(raw_doc_ids))
    assert deduped_doc_ids, "--doc-id requires at least one value"
    assert all(doc_id >= 0 for doc_id in deduped_doc_ids), (
        "--doc-id values must be >= 0"
    )
    return deduped_doc_ids


def main() -> None:
    """Run CLI workflow.

    Args:
        None.

    Returns:
        None.
    """

    args = parse_args()
    config = load_branching_eval_config(config_path=args.config)
    doc_ids = parse_doc_ids(raw_doc_ids=args.doc_id)
    selector_override: SelectorMode | None = None
    if args.selector is not None:
        selector_override = args.selector
    run_dirs = run_experiment_matrix(
        config=config,
        limit=args.limit,
        doc_ids=doc_ids,
        seed_override=args.seed,
        selector_override=selector_override,
        model_override=args.model,
        resume_run_dir=args.resume_run_dir,
    )
    for run_dir in run_dirs:
        print(str(run_dir))


if __name__ == "__main__":
    main()
