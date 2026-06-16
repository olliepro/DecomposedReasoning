from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from random import Random
from types import ModuleType

import numpy as np
import pytest

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = ANALYSIS_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

SCRIPT_PATH = SCRIPTS_ROOT / "plot_steer_exec_faithfulness.py"
SPEC = importlib.util.spec_from_file_location(
    "plot_steer_exec_faithfulness", SCRIPT_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
faith = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = faith
SPEC.loader.exec_module(faith)
assert isinstance(faith, ModuleType)


def test_evenly_spaced_items_includes_endpoints() -> None:
    assert faith.evenly_spaced_items(list(range(10)), count=4) == [0, 3, 6, 9]


def test_extract_clean_pairs_removes_control_tags() -> None:
    text = """
    <think>
      <steer>  Plan <think>quiet</think> next step. </steer>
      <exec> Carry   it out. </exec>
      <steer> </steer><exec>ignored empty steer</exec>
    </think>
    """

    assert faith.extract_clean_pairs(text=text) == [
        ("Plan quiet next step.", "Carry it out.")
    ]


def test_count_wait_instances_matches_literal_word() -> None:
    assert faith.count_wait_instances(text="Wait, wait. await waiting WAIT") == 3


def test_uniform_sampling_is_deterministic() -> None:
    pairs = [
        faith.PairSample(
            run_label="run",
            step=1,
            db_path=Path("db.sqlite"),
            doc_id=index,
            doc_attempt=0,
            leaf_id=f"leaf-{index}",
            leaf_event_index=index,
            pair_index=0,
            steer_text=f"steer {index}",
            exec_text=f"exec {index}",
            steer_wait_count=0,
            exec_wait_count=0,
            wait_count=0,
            answer_reward=0.0,
            raw_answer_acc=False,
            verification=0,
        )
        for index in range(10)
    ]

    first = faith.sample_pairs(pairs=pairs, max_pairs=4, rng=Random(20260614))
    second = faith.sample_pairs(pairs=pairs, max_pairs=4, rng=Random(20260614))

    assert [pair.leaf_id for pair in first] == [pair.leaf_id for pair in second]
    assert len(first) == 4
    assert [pair.doc_id for pair in first] == sorted(pair.doc_id for pair in first)


def test_sqlite_vector_cache_round_trip(tmp_path: Path) -> None:
    cache = faith.EmbeddingCache(path=tmp_path / "embeddings.sqlite")
    vector = np.asarray([0.6, 0.8], dtype=np.float32)

    cache.store_many(model="model", vectors_by_text={"hello": vector})
    lookup = cache.lookup(model="model", texts=["hello", "missing"])

    assert lookup.hit_count == 1
    assert lookup.miss_texts == ["missing"]
    np.testing.assert_array_equal(lookup.vectors_by_text["hello"], vector)


def test_cosine_similarity_fixed_vectors() -> None:
    assert (
        faith.cosine_similarity(
            left=np.asarray([1.0, 0.0], dtype=np.float32),
            right=np.asarray([0.0, 1.0], dtype=np.float32),
        )
        == 0.0
    )
    assert faith.cosine_similarity(
        left=np.asarray([1.0, 1.0], dtype=np.float32),
        right=np.asarray([1.0, 1.0], dtype=np.float32),
    ) == pytest.approx(1.0)


def test_correctness_aggregation_from_leaf_metric(tmp_path: Path) -> None:
    db_dir = tmp_path / "run" / "batch_0001_step_000007"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "tree_events.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("""
            CREATE TABLE leaf_score (
                event_index INTEGER,
                doc_id INTEGER,
                doc_attempt INTEGER,
                leaf_id TEXT,
                verification INTEGER,
                text TEXT,
                stop_reason TEXT
            )
            """)
        connection.execute("""
            CREATE TABLE leaf_metric (
                leaf_event_index INTEGER,
                metric_name TEXT,
                metric_value REAL,
                metric_text TEXT
            )
            """)
        connection.execute(
            """
            INSERT INTO leaf_score
            VALUES (11, 3, 0, 'leaf-a', 1, ?, 'complete')
            """,
            ("<think><steer>Wait first</steer><exec>Do it and wait</exec></think>",),
        )
        connection.execute("""
            INSERT INTO leaf_metric
            VALUES (11, 'answer_reward', 1.0, NULL)
            """)
        connection.execute("""
            INSERT INTO leaf_metric
            VALUES (11, 'raw_answer_acc', 1.0, 'true')
            """)

    step_db = faith.StepDb(
        run_label="run",
        run_root=db_path.parents[1],
        db_path=db_path,
        step=faith.step_from_db_path(db_path=db_path),
        leaf_count=1,
    )

    pairs = faith.read_step_pairs(step_db=step_db)

    assert len(pairs) == 1
    assert pairs[0].answer_reward == 1.0
    assert pairs[0].raw_answer_acc is True
    assert pairs[0].verification == 1
    assert pairs[0].steer_text == "Wait first"
    assert pairs[0].exec_text == "Do it and wait"
    assert pairs[0].steer_wait_count == 1
    assert pairs[0].exec_wait_count == 1
    assert pairs[0].wait_count == 2
