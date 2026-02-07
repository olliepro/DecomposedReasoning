from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to JSONL with one object per line.

    Args:
        path: Output JSONL path.
        rows: Row payloads.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def run_cli_dry_run(repo_dir: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run CLI transform dry-run against a local test output directory.

    Args:
        repo_dir: Repository directory containing `build_sft_dataset.py`.
        output_dir: Stage output directory passed to `--output-dir`.

    Returns:
        Completed subprocess result.

    Example:
        >>> # run_cli_dry_run(repo_dir=Path('.'), output_dir=Path('tmp/output'))
        >>> # doctest: +SKIP
    """
    return subprocess.run(
        [
            sys.executable,
            str(repo_dir / "build_sft_dataset.py"),
            "--stage",
            "transform",
            "--dry-run",
            "--yes",
            "--max-rows",
            "2",
            "--output-dir",
            str(output_dir),
            "--env-file",
            str(repo_dir / ".env"),
            "--system-prompt",
            str(repo_dir / "system_prompt.md"),
            "--user-prompt",
            str(repo_dir / "user_prompt.md"),
        ],
        cwd=str(repo_dir),
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    """Run a dry-run CLI smoke test without sending API requests."""
    repo_dir = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="buildsft_dryrun_") as tmpdir:
        output_dir = Path(tmpdir) / "output"
        write_jsonl(
            path=output_dir / "stratified_sample.jsonl",
            rows=[
                {
                    "id": "row-1",
                    "dataset_source": "math",
                    "messages": [
                        {"role": "user", "content": "Solve x^2=4."},
                        {
                            "role": "assistant",
                            "content": "<think>factor x^2-4</think>\nanswer: x=+/-2",
                        },
                    ],
                },
                {
                    "id": "row-2",
                    "dataset_source": "math",
                    "messages": [
                        {"role": "user", "content": "Integrate x dx."},
                        {
                            "role": "assistant",
                            "content": "<think>power rule</think>\nanswer: x^2/2 + C",
                        },
                    ],
                },
            ],
        )
        result = run_cli_dry_run(repo_dir=repo_dir, output_dir=output_dir)
        assert result.returncode == 0, result.stderr
        assert "Running stage: transform" in result.stdout
        assert '"rows_selected": 2' in result.stdout
        assert '"batch": true' in result.stdout
    print("ok: dry-run transform smoke test passed")


if __name__ == "__main__":
    main()
