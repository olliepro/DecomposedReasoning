"""Plot steer/exec embedding similarity from RL step SQLite DBs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import Iterable, Sequence, TypeVar
from urllib import error, request

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from branching_eval.embedding_selection import resolve_openai_api_key  # noqa: E402

STEP_RE = re.compile(r"batch_(?P<batch>\d+)_step_(?P<step>\d+)$")
PAIR_RE = re.compile(
    r"<steer\b[^>]*>(?P<steer>.*?)</steer>\s*" r"<exec\b[^>]*>(?P<exec>.*?)</exec>",
    flags=re.IGNORECASE | re.DOTALL,
)
TAG_RE = re.compile(r"</?(?:think|steer|exec)\b[^>]*>", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
WAIT_RE = re.compile(r"\bwait\b", flags=re.IGNORECASE)

DEFAULT_RUN_ROOTS = (
    Path(
        "/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/RLTraining/"
        "qwen35_branching_dapo/artifacts/"
        "qwen35_4b_branch_gs50_branching_gs50_branch_all_lr3e6_steer30_"
        "5553508_20260612T133359Z"
    ),
    Path(
        "/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/RLTraining/"
        "qwen35_branching_dapo/artifacts/"
        "qwen35_4b_branch_gs50_structured_baseline_gs50_struct_all_steer30_"
        "5536458_20260611T062724Z"
    ),
)
DEFAULT_OUTPUT_BASE = Path(
    "/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/RLTraining/"
    "qwen35_branching_dapo/faithfulness"
)
DEFAULT_CACHE_BASE = Path(
    "/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/RLTraining/"
    "qwen35_branching_dapo/faithfulness_cache"
)
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 1024
DEFAULT_SEED = 20260614
DEFAULT_STEP_COUNT = 25
DEFAULT_PAIRS_PER_STEP = 200
RETRY_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
T = TypeVar("T")


@dataclass(frozen=True)
class StepDb:
    """One scored step DB."""

    run_label: str
    run_root: Path
    db_path: Path
    step: int
    leaf_count: int


@dataclass(frozen=True)
class PairSample:
    """One sampled steer/exec pair and its source trajectory metadata."""

    run_label: str
    step: int
    db_path: Path
    doc_id: int
    doc_attempt: int
    leaf_id: str
    leaf_event_index: int
    pair_index: int
    steer_text: str
    exec_text: str
    steer_wait_count: int
    exec_wait_count: int
    wait_count: int
    answer_reward: float
    raw_answer_acc: bool
    verification: int
    similarity: float | None = None


@dataclass(frozen=True)
class StepSummary:
    """Per-step aggregate plotted by the CLI."""

    run_label: str
    step: int
    pair_count: int
    correct_pair_count: int
    incorrect_pair_count: int
    mean_similarity: float
    similarity_stderr: float
    mean_wait_count: float
    wait_count_stderr: float
    total_wait_count: int
    mean_answer_reward: float
    correct_mean_similarity: float | None
    incorrect_mean_similarity: float | None


@dataclass(frozen=True)
class CacheLookup:
    """Cache lookup result and hit/miss counts."""

    vectors_by_text: dict[str, np.ndarray]
    hit_count: int
    miss_texts: list[str]


class EmbeddingCache:
    """Day-scoped SQLite vector cache using float32 BLOBs."""

    def __init__(self, *, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    model TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_float32_blob BLOB NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    PRIMARY KEY (model, text_sha256)
                )
                """)

    def lookup(self, *, model: str, texts: Sequence[str]) -> CacheLookup:
        vectors_by_text: dict[str, np.ndarray] = {}
        miss_texts: list[str] = []
        hit_count = 0
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as connection:
            for text in texts:
                row = connection.execute(
                    """
                    SELECT dimensions, vector_float32_blob
                    FROM embedding_cache
                    WHERE model = ? AND text_sha256 = ?
                    """,
                    (model, text_sha256(text=text)),
                ).fetchone()
                if row is None:
                    miss_texts.append(text)
                    continue
                vector = np.frombuffer(row[1], dtype=np.float32).copy()
                assert vector.shape == (int(row[0]),)
                vectors_by_text[text] = vector
                hit_count += 1
        return CacheLookup(
            vectors_by_text=vectors_by_text,
            hit_count=hit_count,
            miss_texts=miss_texts,
        )

    def store_many(self, *, model: str, vectors_by_text: dict[str, np.ndarray]) -> None:
        now = datetime.now(UTC).isoformat()
        rows = [
            (
                model,
                text_sha256(text=text),
                int(vector.shape[0]),
                np.asarray(vector, dtype=np.float32).tobytes(),
                now,
            )
            for text, vector in vectors_by_text.items()
        ]
        with sqlite3.connect(self.path) as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache (
                    model, text_sha256, dimensions, vector_float32_blob, created_at_utc
                ) VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )


def text_sha256(*, text: str) -> str:
    """Return a stable text hash for cache keys."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", type=Path, default=[])
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--cache-base", type=Path, default=DEFAULT_CACHE_BASE)
    parser.add_argument("--date-stamp", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--step-count", type=int, default=DEFAULT_STEP_COUNT)
    parser.add_argument("--pairs-per-step", type=int, default=DEFAULT_PAIRS_PER_STEP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--fake-embeddings", action="store_true")
    parser.add_argument("--fake-dimensions", type=int, default=64)
    parser.add_argument("--max-steps-per-run", type=int, default=None)
    parser.add_argument("--max-pairs-per-step", type=int, default=None)
    return parser.parse_args()


def step_from_db_path(*, db_path: Path) -> int:
    """Extract trainer step from a step DB path."""

    match = STEP_RE.fullmatch(db_path.parent.name)
    assert match is not None, f"unexpected step DB parent: {db_path.parent}"
    return int(match.group("step"))


def run_label_from_root(*, run_root: Path) -> str:
    """Return a compact label for the known Qwen35 run families."""

    name = run_root.name
    if "_branching_" in name:
        return "branching_all_lr3e-6"
    if "_structured_baseline_" in name:
        return "structured_all_lr1e-6"
    return name


def collect_scored_step_dbs(*, run_root: Path) -> list[StepDb]:
    """Collect non-empty step DBs with scored leaves."""

    run_label = run_label_from_root(run_root=run_root)
    step_dbs: list[StepDb] = []
    for db_path in sorted(run_root.glob("batch_*_step_*/tree_events.sqlite")):
        if db_path.stat().st_size <= 0:
            continue
        leaf_count = read_leaf_count(db_path=db_path)
        if leaf_count <= 0:
            continue
        step_dbs.append(
            StepDb(
                run_label=run_label,
                run_root=run_root,
                db_path=db_path,
                step=step_from_db_path(db_path=db_path),
                leaf_count=leaf_count,
            )
        )
    return sorted(step_dbs, key=lambda item: item.step)


def read_leaf_count(*, db_path: Path) -> int:
    """Return scored leaf count from one DB."""

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM leaf_score").fetchone()[0])


def evenly_spaced_items(items: Sequence[T], *, count: int) -> list[T]:
    """Return up to count evenly spaced items including endpoints."""

    assert count >= 1, "count must be positive"
    if len(items) <= count:
        return list(items)
    indexes = sorted(
        {round(index * (len(items) - 1) / (count - 1)) for index in range(count)}
    )
    return [items[index] for index in indexes]


def clean_embedding_text(*, text: str) -> str:
    """Strip control tags and normalize whitespace before embedding."""

    without_tags = TAG_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", without_tags).strip()


def extract_clean_pairs(*, text: str) -> list[tuple[str, str]]:
    """Extract valid cleaned steer/exec pairs from one leaf text."""

    pairs: list[tuple[str, str]] = []
    for match in PAIR_RE.finditer(text):
        steer_text = clean_embedding_text(text=match.group("steer"))
        exec_text = clean_embedding_text(text=match.group("exec"))
        if steer_text and exec_text:
            pairs.append((steer_text, exec_text))
    return pairs


def count_wait_instances(*, text: str) -> int:
    """Count literal whole-word wait instances in cleaned text."""

    return len(WAIT_RE.findall(text))


def read_step_pairs(*, step_db: StepDb) -> list[PairSample]:
    """Read all valid steer/exec pair samples from one step DB."""

    pairs: list[PairSample] = []
    with sqlite3.connect(f"file:{step_db.db_path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(LEAF_QUERY):
            pairs.extend(pair_samples_from_leaf(step_db=step_db, row=row))
    return pairs


LEAF_QUERY = """
SELECT
    leaf_score.event_index,
    leaf_score.doc_id,
    leaf_score.doc_attempt,
    leaf_score.leaf_id,
    leaf_score.verification,
    leaf_score.text,
    leaf_score.stop_reason,
    MAX(CASE WHEN leaf_metric.metric_name = 'answer_reward'
        THEN leaf_metric.metric_value END) AS answer_reward,
    MAX(CASE WHEN leaf_metric.metric_name = 'raw_answer_acc'
        THEN leaf_metric.metric_value END) AS raw_answer_acc_value,
    MAX(CASE WHEN leaf_metric.metric_name = 'raw_answer_acc'
        THEN leaf_metric.metric_text END) AS raw_answer_acc_text
FROM leaf_score
LEFT JOIN leaf_metric
  ON leaf_metric.leaf_event_index = leaf_score.event_index
GROUP BY leaf_score.event_index
ORDER BY leaf_score.event_index
"""


def pair_samples_from_leaf(*, step_db: StepDb, row: sqlite3.Row) -> list[PairSample]:
    """Convert one scored leaf row into pair samples."""

    answer_reward = metric_float(value=row["answer_reward"])
    raw_answer_acc = metric_bool(
        value=row["raw_answer_acc_value"],
        text=row["raw_answer_acc_text"],
    )
    pairs = extract_clean_pairs(text=str(row["text"] or ""))
    return [
        PairSample(
            run_label=step_db.run_label,
            step=step_db.step,
            db_path=step_db.db_path,
            doc_id=int(row["doc_id"]),
            doc_attempt=int(row["doc_attempt"]),
            leaf_id=str(row["leaf_id"]),
            leaf_event_index=int(row["event_index"]),
            pair_index=pair_index,
            steer_text=steer_text,
            exec_text=exec_text,
            steer_wait_count=count_wait_instances(text=steer_text),
            exec_wait_count=count_wait_instances(text=exec_text),
            wait_count=count_wait_instances(text=steer_text)
            + count_wait_instances(text=exec_text),
            answer_reward=answer_reward,
            raw_answer_acc=raw_answer_acc,
            verification=int(row["verification"]),
        )
        for pair_index, (steer_text, exec_text) in enumerate(pairs)
    ]


def metric_float(*, value: object) -> float:
    """Return a nullable SQLite metric value as float."""

    if value is None:
        return 0.0
    assert isinstance(value, int | float | str)
    return float(value)


def metric_bool(*, value: object, text: object) -> bool:
    """Return a nullable SQLite metric as bool."""

    if value is not None:
        assert isinstance(value, int | float | str)
        return float(value) != 0.0
    return str(text or "").strip().lower() in {"true", "1", "1.0", "yes", "y"}


def sample_pairs(
    *, pairs: Sequence[PairSample], max_pairs: int, rng: Random
) -> list[PairSample]:
    """Uniformly sample pairs from one step."""

    assert max_pairs >= 1, "max_pairs must be positive"
    if len(pairs) <= max_pairs:
        return list(pairs)
    indexes = sorted(rng.sample(range(len(pairs)), k=max_pairs))
    return [pairs[index] for index in indexes]


def unique_texts_from_pairs(*, pairs: Sequence[PairSample]) -> list[str]:
    """Return unique steer/exec texts in first-seen order."""

    return list(
        dict.fromkeys(
            text for pair in pairs for text in (pair.steer_text, pair.exec_text)
        )
    )


def resolve_embeddings(
    *,
    texts: Sequence[str],
    cache: EmbeddingCache,
    model: str,
    batch_size: int,
    fake_embeddings: bool,
    fake_dimensions: int,
) -> tuple[dict[str, np.ndarray], int, int]:
    """Resolve embeddings from cache and API or fake deterministic vectors."""

    unique_texts = list(dict.fromkeys(texts))
    lookup = cache.lookup(model=model, texts=unique_texts)
    if fake_embeddings:
        fetched = fake_embeddings_by_text(
            texts=lookup.miss_texts,
            dimensions=fake_dimensions,
        )
    else:
        api_key = resolve_openai_api_key(
            env_paths=(ANALYSIS_ROOT / ".env", Path(".env"))
        )
        assert (
            api_key is not None
        ), "OPENAI_API_KEY is required unless --fake-embeddings is used"
        fetched = fetch_openai_embeddings(
            texts=lookup.miss_texts,
            api_key=api_key,
            model=model,
            batch_size=batch_size,
        )
    cache.store_many(model=model, vectors_by_text=fetched)
    return (
        {**lookup.vectors_by_text, **fetched},
        lookup.hit_count,
        len(lookup.miss_texts),
    )


def fake_embeddings_by_text(
    *, texts: Sequence[str], dimensions: int
) -> dict[str, np.ndarray]:
    """Return deterministic fake vectors for dry runs and tests."""

    assert dimensions >= 2, "fake dimensions must be at least 2"
    return {text: fake_embedding(text=text, dimensions=dimensions) for text in texts}


def fake_embedding(*, text: str, dimensions: int) -> np.ndarray:
    """Build one deterministic non-zero fake embedding vector."""

    values: list[float] = []
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    vector = np.asarray(values[:dimensions], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    assert norm > 0.0, "fake vector must be non-zero"
    return vector / norm


def fetch_openai_embeddings(
    *, texts: Sequence[str], api_key: str, model: str, batch_size: int
) -> dict[str, np.ndarray]:
    """Fetch OpenAI embeddings in large array-input batches."""

    vectors: dict[str, np.ndarray] = {}
    for batch_index, batch in enumerate(
        batch_texts(texts=texts, batch_size=batch_size), start=1
    ):
        print(f"embedding batch {batch_index}: texts={len(batch)}")
        batch_vectors = fetch_openai_embedding_batch(
            texts=batch,
            api_key=api_key,
            model=model,
        )
        vectors.update(zip(batch, batch_vectors, strict=True))
    return vectors


def batch_texts(*, texts: Sequence[str], batch_size: int) -> list[list[str]]:
    """Split texts into API request batches."""

    assert batch_size >= 1, "batch_size must be positive"
    return [
        list(texts[index : index + batch_size])
        for index in range(0, len(texts), batch_size)
    ]


def fetch_openai_embedding_batch(
    *, texts: Sequence[str], api_key: str, model: str
) -> list[np.ndarray]:
    """Fetch one embedding batch with retries."""

    payload = json.dumps(
        {"model": model, "input": list(texts), "encoding_format": "float"}
    ).encode()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(1, 6):
        try:
            req = request.Request(
                "https://api.openai.com/v1/embeddings",
                data=payload,
                headers=headers,
                method="POST",
            )
            with request.urlopen(req, timeout=300) as response:
                return parse_embedding_response(payload=json.loads(response.read()))
        except error.HTTPError as exc:
            if exc.code not in RETRY_STATUS_CODES or attempt == 5:
                raise RuntimeError(
                    openai_error_message(exc=exc, attempt=attempt)
                ) from exc
            time.sleep(min(40.0, 5.0 * (2 ** (attempt - 1))))
        except error.URLError as exc:
            if attempt == 5:
                raise RuntimeError(
                    f"OpenAI embedding transport error after {attempt} attempts: {exc}"
                ) from exc
            time.sleep(min(40.0, 5.0 * (2 ** (attempt - 1))))
    raise AssertionError("unreachable embedding retry state")


def parse_embedding_response(*, payload: dict[str, object]) -> list[np.ndarray]:
    """Parse vectors from an OpenAI embeddings response."""

    data = payload.get("data")
    assert isinstance(data, list), "embedding response missing data"
    rows = [item for item in data if isinstance(item, dict)]
    ordered = sorted(rows, key=lambda item: int(item["index"]))
    return [np.asarray(item["embedding"], dtype=np.float32) for item in ordered]


def openai_error_message(*, exc: error.HTTPError, attempt: int) -> str:
    """Return a bounded OpenAI HTTP error message."""

    body = exc.read().decode("utf-8", errors="replace")[:2000]
    return f"OpenAI embedding error {exc.code} after {attempt} attempts: {body}"


def score_pairs(
    *, pairs: Sequence[PairSample], embeddings_by_text: dict[str, np.ndarray]
) -> list[PairSample]:
    """Attach cosine similarities to sampled pairs."""

    return [
        replace(
            pair,
            similarity=cosine_similarity(
                left=embeddings_by_text[pair.steer_text],
                right=embeddings_by_text[pair.exec_text],
            ),
        )
        for pair in pairs
    ]


def cosine_similarity(*, left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity for two embedding vectors."""

    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    assert left_norm > 0.0 and right_norm > 0.0, "vectors must be non-zero"
    return float(np.dot(left, right) / (left_norm * right_norm))


def summarize_steps(*, pairs: Sequence[PairSample]) -> list[StepSummary]:
    """Aggregate scored pairs by run and step."""

    grouped: dict[tuple[str, int], list[PairSample]] = {}
    for pair in pairs:
        grouped.setdefault((pair.run_label, pair.step), []).append(pair)
    return [
        summarize_group(run_label=run_label, step=step, pairs=group_pairs)
        for (run_label, step), group_pairs in sorted(grouped.items())
    ]


def summarize_group(
    *, run_label: str, step: int, pairs: Sequence[PairSample]
) -> StepSummary:
    """Aggregate one run-step group."""

    similarities = np.asarray(
        [assert_similarity(pair=pair) for pair in pairs], dtype=np.float64
    )
    answer_rewards = np.asarray(
        [pair.answer_reward for pair in pairs], dtype=np.float64
    )
    wait_counts = np.asarray([pair.wait_count for pair in pairs], dtype=np.float64)
    correct = [pair for pair in pairs if pair.answer_reward > 0.0]
    incorrect = [pair for pair in pairs if pair.answer_reward <= 0.0]
    return StepSummary(
        run_label=run_label,
        step=step,
        pair_count=len(pairs),
        correct_pair_count=len(correct),
        incorrect_pair_count=len(incorrect),
        mean_similarity=float(np.mean(similarities)),
        similarity_stderr=standard_error(values=similarities),
        mean_wait_count=float(np.mean(wait_counts)),
        wait_count_stderr=standard_error(values=wait_counts),
        total_wait_count=int(np.sum(wait_counts)),
        mean_answer_reward=float(np.mean(answer_rewards)),
        correct_mean_similarity=optional_similarity_mean(pairs=correct),
        incorrect_mean_similarity=optional_similarity_mean(pairs=incorrect),
    )


def assert_similarity(*, pair: PairSample) -> float:
    """Return present similarity value."""

    assert pair.similarity is not None, "pair has not been scored"
    return pair.similarity


def standard_error(*, values: np.ndarray) -> float:
    """Return sample standard error, or zero for one value."""

    if values.shape[0] <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.shape[0]))


def optional_similarity_mean(*, pairs: Sequence[PairSample]) -> float | None:
    """Return mean similarity for non-empty pair groups."""

    if not pairs:
        return None
    return float(np.mean([assert_similarity(pair=pair) for pair in pairs]))


def pearson_correlation(*, xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Return Pearson correlation, or None when undefined."""

    if len(xs) < 2 or len(ys) < 2:
        return None
    x_values = np.asarray(xs, dtype=np.float64)
    y_values = np.asarray(ys, dtype=np.float64)
    if float(np.std(x_values)) == 0.0 or float(np.std(y_values)) == 0.0:
        return None
    return float(np.corrcoef(x_values, y_values)[0, 1])


def plot_similarity_over_steps(
    *, summaries: Sequence[StepSummary], output_path: Path
) -> None:
    """Plot mean steer/exec similarity over trainer steps."""

    fig, ax = plt.subplots(figsize=(13, 7))
    for run_label in sorted({summary.run_label for summary in summaries}):
        rows = [summary for summary in summaries if summary.run_label == run_label]
        steps = np.asarray([row.step for row in rows])
        means = np.asarray([row.mean_similarity for row in rows])
        stderrs = np.asarray([row.similarity_stderr for row in rows])
        ax.plot(steps, means, marker="o", linewidth=2, label=run_label)
        ax.fill_between(steps, means - stderrs, means + stderrs, alpha=0.18)
    ax.set_title("Steer/Exec Embedding Similarity Over RL Steps")
    ax.set_xlabel("Trainer step")
    ax.set_ylabel("Mean cosine similarity")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    save_figure(fig=fig, output_path=output_path)


def plot_similarity_vs_correctness(
    *, summaries: Sequence[StepSummary], output_path: Path
) -> None:
    """Plot step mean similarity against step mean correctness."""

    fig, ax = plt.subplots(figsize=(10, 7))
    for run_label in sorted({summary.run_label for summary in summaries}):
        rows = [summary for summary in summaries if summary.run_label == run_label]
        similarities = [row.mean_similarity for row in rows]
        rewards = [row.mean_answer_reward for row in rows]
        corr = pearson_correlation(xs=similarities, ys=rewards)
        label = f"{run_label} r={format_corr(corr=corr)}"
        ax.scatter(similarities, rewards, s=46, alpha=0.8, label=label)
    ax.set_title("Step Similarity vs Answer Correctness")
    ax.set_xlabel("Mean steer/exec cosine similarity")
    ax.set_ylabel("Mean answer_reward")
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    save_figure(fig=fig, output_path=output_path)


def plot_similarity_by_correctness(
    *, summaries: Sequence[StepSummary], output_path: Path
) -> None:
    """Plot correct and incorrect trajectory similarity over steps."""

    fig, ax = plt.subplots(figsize=(13, 7))
    for run_label in sorted({summary.run_label for summary in summaries}):
        rows = [summary for summary in summaries if summary.run_label == run_label]
        steps = [row.step for row in rows]
        correct = [
            (
                np.nan
                if row.correct_mean_similarity is None
                else row.correct_mean_similarity
            )
            for row in rows
        ]
        incorrect = [
            (
                np.nan
                if row.incorrect_mean_similarity is None
                else row.incorrect_mean_similarity
            )
            for row in rows
        ]
        ax.plot(steps, correct, marker="o", linewidth=2, label=f"{run_label} correct")
        ax.plot(
            steps,
            incorrect,
            marker="x",
            linestyle="--",
            linewidth=1.6,
            label=f"{run_label} incorrect",
        )
    ax.set_title("Steer/Exec Similarity by Source Trajectory Correctness")
    ax.set_xlabel("Trainer step")
    ax.set_ylabel("Mean cosine similarity")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    save_figure(fig=fig, output_path=output_path)


def plot_wait_instances_over_steps(
    *, summaries: Sequence[StepSummary], output_path: Path
) -> None:
    """Plot literal wait instances per sampled steer/exec pair over steps."""

    fig, ax = plt.subplots(figsize=(13, 7))
    for run_label in sorted({summary.run_label for summary in summaries}):
        rows = [summary for summary in summaries if summary.run_label == run_label]
        steps = np.asarray([row.step for row in rows])
        means = np.asarray([row.mean_wait_count for row in rows])
        stderrs = np.asarray([row.wait_count_stderr for row in rows])
        ax.plot(steps, means, marker="o", linewidth=2, label=run_label)
        ax.fill_between(steps, means - stderrs, means + stderrs, alpha=0.18)
    ax.set_title('Whole-Word "wait" Instances Over RL Steps')
    ax.set_xlabel("Trainer step")
    ax.set_ylabel('Mean "wait" instances per sampled pair')
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    save_figure(fig=fig, output_path=output_path)


def save_figure(*, fig: Figure, output_path: Path) -> None:
    """Save and close a matplotlib figure."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def format_corr(*, corr: float | None) -> str:
    """Return printable correlation text."""

    return "n/a" if corr is None else f"{corr:.3f}"


def print_run_correlations(
    *, summaries: Sequence[StepSummary], pairs: Sequence[PairSample]
) -> None:
    """Print pair-level and step-level correctness correlations."""

    for run_label in sorted({pair.run_label for pair in pairs}):
        run_pairs = [pair for pair in pairs if pair.run_label == run_label]
        run_summaries = [
            summary for summary in summaries if summary.run_label == run_label
        ]
        pair_corr = pearson_correlation(
            xs=[assert_similarity(pair=pair) for pair in run_pairs],
            ys=[pair.answer_reward for pair in run_pairs],
        )
        step_corr = pearson_correlation(
            xs=[summary.mean_similarity for summary in run_summaries],
            ys=[summary.mean_answer_reward for summary in run_summaries],
        )
        print(
            f"{run_label}: pair_corr={format_corr(corr=pair_corr)} "
            f"step_corr={format_corr(corr=step_corr)}"
        )


def print_run_wait_counts(*, summaries: Sequence[StepSummary]) -> None:
    """Print sampled wait counts by run."""

    for run_label in sorted({summary.run_label for summary in summaries}):
        run_summaries = [
            summary for summary in summaries if summary.run_label == run_label
        ]
        total_wait_count = sum(summary.total_wait_count for summary in run_summaries)
        total_pairs = sum(summary.pair_count for summary in run_summaries)
        mean_wait_count = total_wait_count / total_pairs
        print(
            f"{run_label}: wait_count={total_wait_count} "
            f"wait_per_pair={mean_wait_count:.4f}"
        )


def selected_step_dbs_for_roots(
    *, run_roots: Sequence[Path], step_count: int, max_steps_per_run: int | None
) -> list[StepDb]:
    """Collect evenly spaced scored step DBs across run roots."""

    selected: list[StepDb] = []
    for run_root in run_roots:
        scored = collect_scored_step_dbs(run_root=run_root)
        if max_steps_per_run is not None:
            scored = scored[:max_steps_per_run]
        chosen = evenly_spaced_items(items=scored, count=step_count)
        print(
            f"{run_label_from_root(run_root=run_root)}: scored_steps={len(scored)} "
            f"selected={[step_db.step for step_db in chosen]}"
        )
        selected.extend(chosen)
    return selected


def collect_sampled_pairs(
    *, step_dbs: Sequence[StepDb], pairs_per_step: int, seed: int
) -> list[PairSample]:
    """Read and uniformly sample pairs from selected step DBs."""

    rng = Random(seed)
    sampled: list[PairSample] = []
    for step_db in step_dbs:
        step_pairs = read_step_pairs(step_db=step_db)
        chosen = sample_pairs(pairs=step_pairs, max_pairs=pairs_per_step, rng=rng)
        sampled.extend(chosen)
        print(
            f"{step_db.run_label} step={step_db.step}: "
            f"valid_pairs={len(step_pairs)} sampled={len(chosen)}"
        )
    return sampled


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    run_roots = tuple(args.run_root) if args.run_root else DEFAULT_RUN_ROOTS
    output_dir = args.output_base / args.date_stamp
    cache_path = args.cache_base / args.date_stamp / "embeddings.sqlite"
    pairs_per_step = args.max_pairs_per_step or args.pairs_per_step
    step_dbs = selected_step_dbs_for_roots(
        run_roots=run_roots,
        step_count=args.step_count,
        max_steps_per_run=args.max_steps_per_run,
    )
    pairs = collect_sampled_pairs(
        step_dbs=step_dbs,
        pairs_per_step=pairs_per_step,
        seed=args.seed,
    )
    assert pairs, "no steer/exec pairs sampled"
    cache = EmbeddingCache(path=cache_path)
    embeddings, hit_count, miss_count = resolve_embeddings(
        texts=unique_texts_from_pairs(pairs=pairs),
        cache=cache,
        model=args.model,
        batch_size=args.batch_size,
        fake_embeddings=args.fake_embeddings,
        fake_dimensions=args.fake_dimensions,
    )
    scored_pairs = score_pairs(pairs=pairs, embeddings_by_text=embeddings)
    summaries = summarize_steps(pairs=scored_pairs)
    plot_similarity_over_steps(
        summaries=summaries,
        output_path=output_dir / "steer_exec_similarity_over_steps.png",
    )
    plot_similarity_vs_correctness(
        summaries=summaries,
        output_path=output_dir / "steer_exec_similarity_vs_correctness.png",
    )
    plot_similarity_by_correctness(
        summaries=summaries,
        output_path=output_dir / "steer_exec_similarity_by_correctness_over_steps.png",
    )
    plot_wait_instances_over_steps(
        summaries=summaries,
        output_path=output_dir / "wait_instances_over_steps.png",
    )
    print(f"cache={cache_path} hits={hit_count} misses={miss_count}")
    print(f"sampled_pairs={len(scored_pairs)} output_dir={output_dir}")
    print_run_correlations(summaries=summaries, pairs=scored_pairs)
    print_run_wait_counts(summaries=summaries)


if __name__ == "__main__":
    main()
