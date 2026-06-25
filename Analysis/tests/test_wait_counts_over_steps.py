from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import pytest

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = ANALYSIS_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

SCRIPT_PATH = SCRIPTS_ROOT / "plot_wait_counts_over_steps.py"
SPEC = importlib.util.spec_from_file_location(
    "plot_wait_counts_over_steps", SCRIPT_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
wait_plot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wait_plot
SPEC.loader.exec_module(wait_plot)
assert isinstance(wait_plot, ModuleType)


def test_count_wait_instances_handles_punctuation() -> None:
    text = 'Wait, wait. "WAIT"! await waiting waited wait_time wait2'

    assert wait_plot.count_wait_instances(text=text) == 3


def test_exec_block_char_count() -> None:
    text = "<exec>abc</exec><steer>x</steer><exec>de\n</exec>"

    assert wait_plot.exec_block_char_count(text=text) == (2, 6)


def test_summarize_texts_counts_wait_and_exec_lengths() -> None:
    stats = wait_plot.summarize_texts(
        step=5,
        texts=[
            "<think><exec>Wait, compute.</exec></think>",
            "<think><exec>do not wait</exec><exec>done</exec></think>",
        ],
    )

    assert stats.step == 5
    assert stats.leaf_count == 2
    assert stats.wait_count == 2
    assert stats.exec_block_count == 3
    assert stats.exec_block_chars == 29
    assert stats.mean_exec_block_chars == pytest.approx(29 / 3)
    assert stats.wait_per_million_chars == pytest.approx(
        (2 / stats.total_chars) * 1_000_000.0
    )


def test_collect_step_dbs_skips_empty_leaf_scores(tmp_path: Path) -> None:
    empty_db = tmp_path / "batch_0000_step_000001" / "tree_events.sqlite"
    full_db = tmp_path / "batch_0001_step_000002" / "tree_events.sqlite"
    empty_db.parent.mkdir()
    full_db.parent.mkdir()
    create_leaf_score_db(db_path=empty_db, texts=[])
    create_leaf_score_db(db_path=full_db, texts=["<exec>wait</exec>"])

    step_dbs = wait_plot.collect_step_dbs(run_root=tmp_path)

    assert [(step_db.step, step_db.leaf_count) for step_db in step_dbs] == [(2, 1)]


def create_leaf_score_db(*, db_path: Path, texts: list[str]) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute("""
            CREATE TABLE leaf_score (
                event_index INTEGER PRIMARY KEY,
                text TEXT NOT NULL
            )
            """)
        for event_index, text in enumerate(texts):
            connection.execute(
                "INSERT INTO leaf_score (event_index, text) VALUES (?, ?)",
                (event_index, text),
            )
