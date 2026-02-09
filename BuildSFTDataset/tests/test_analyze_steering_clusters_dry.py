from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_steering_clusters import (  # noqa: E402
    ClusterLabelRecord,
    RetryConfig,
    RunConfig,
    build_unique_text_index,
    cluster_centroid_indices,
    cluster_texts_using_unique_strings,
    embed_texts_async,
    hash_text,
    parse_steering_execution_pairs,
    stage_assign_noise,
    stage_cluster1_async,
    stage_cluster2_async,
    stage_extract,
    stage_name1_async,
    stage_name2_async,
    stage_report,
    stage_tokens_async,
)


@dataclass(frozen=True)
class FakeTokenResponse:
    """Mock count_tokens response.

    Args:
        total_tokens: Token count integer.
    """

    total_tokens: int


@dataclass(frozen=True)
class FakeEmbedding:
    """Mock embedding item.

    Args:
        values: Embedding vector values.
    """

    values: list[float]


@dataclass(frozen=True)
class FakeEmbedResponse:
    """Mock embed_content response.

    Args:
        embeddings: Embedding list.
    """

    embeddings: list[FakeEmbedding]


@dataclass(frozen=True)
class FakeGenerateResponse:
    """Mock generate_content response.

    Args:
        text: JSON response text.
    """

    text: str


class FakeAsyncModels:
    """Async Gemini models mock for dry integration tests."""

    async def count_tokens(
        self, *, model: str, contents: str, config: object | None = None
    ) -> FakeTokenResponse:
        """Return token count proxy based on whitespace tokens.

        Args:
            model: Model name.
            contents: Input text.
            config: Optional config.

        Returns:
            Fake token response.
        """

        del model, config
        return FakeTokenResponse(total_tokens=len(str(contents).split()))

    async def embed_content(
        self, *, model: str, contents: object, config: object | None = None
    ) -> FakeEmbedResponse:
        """Return deterministic low-dimensional embeddings.

        Args:
            model: Embedding model.
            contents: Input text.
            config: Optional config.

        Returns:
            Fake embedding response.
        """

        del model, config

        def _embed_single_text(text_value: str) -> list[float]:
            lowered_text = text_value.lower()
            if "math" in lowered_text:
                return [1.0, 0.0, 0.0, 0.0]
            if "code" in lowered_text:
                return [0.0, 1.0, 0.0, 0.0]
            digest = hashlib.sha256(lowered_text.encode("utf-8")).digest()
            return [float(digest[i]) / 255.0 for i in range(4)]

        if isinstance(contents, list):
            vectors = [
                FakeEmbedding(values=_embed_single_text(text_value=str(item)))
                for item in contents
            ]
            return FakeEmbedResponse(embeddings=vectors)

        return FakeEmbedResponse(
            embeddings=[
                FakeEmbedding(values=_embed_single_text(text_value=str(contents)))
            ]
        )

    async def generate_content(
        self, *, model: str, contents: str, config: object | None = None
    ) -> FakeGenerateResponse:
        """Return deterministic cluster naming JSON.

        Args:
            model: Naming model.
            contents: Prompt text.
            config: Optional config.

        Returns:
            Fake generate response.
        """

        del model, config
        text = str(contents)
        if "math" in text.lower():
            payload = {
                "title": "Math Reasoning",
                "purpose": "Solve numeric/math tasks",
                "keywords": ["math", "algebra", "proof"],
            }
        elif "code" in text.lower():
            payload = {
                "title": "Code Reasoning",
                "purpose": "Develop or debug code",
                "keywords": ["code", "python", "debug"],
            }
        else:
            payload = {
                "title": "General Reasoning",
                "purpose": "General cluster",
                "keywords": ["general", "reasoning", "analysis"],
            }
        return FakeGenerateResponse(text=json.dumps(payload))


class FakeAio:
    """Mock aio wrapper with models endpoint."""

    def __init__(self) -> None:
        self.models = FakeAsyncModels()


class FakeClient:
    """Mock Gemini client exposing aio models."""

    def __init__(self) -> None:
        self.aio = FakeAio()


def build_test_config(
    base_dir: Path, transformed_path: Path, og_path: Path
) -> RunConfig:
    """Create test config for dry pipeline execution.

    Args:
        base_dir: Output base directory.
        transformed_path: Transformed dataset path.
        og_path: OG dataset path.

    Returns:
        RunConfig instance.
    """

    return RunConfig(
        transformed_path=transformed_path,
        og_path=og_path,
        output_dir=base_dir,
        env_file=base_dir / ".env",
        naming_model="gemini-3-flash-preview",
        embedding_model="gemini-embedding-001",
        min_cluster_size=2,
        max_cluster_size=2000,
        naming_concurrency=10,
        embed_concurrency=4,
        token_concurrency=4,
        embed_batch_size=100,
        naming_requests_per_minute=300,
        embed_requests_per_minute=300,
        token_requests_per_minute=300,
        tsne_sample_size=200,
        seed=7,
        resume=False,
        stage="all",
        api_timeout_seconds=15,
        max_retries=2,
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write JSONL rows to path.

    Args:
        path: Output path.
        rows: Row payloads.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def test_parse_steering_execution_pairs_mixed_tags() -> None:
    """Verify parser supports mixed steer/steering and exec/execute/execution tags."""

    think_text = (
        "<steer>A</steer><exec>B</exec><steering>C</steering><execution>D</execution>"
    )
    pairs, meta = parse_steering_execution_pairs(think_text=think_text)
    assert pairs == [("A", "B"), ("C", "D")]
    assert meta["paired_count"] == 2


def test_cluster_centroid_indices_max_20_and_deterministic() -> None:
    """Ensure centroid sample selector is deterministic and capped to 20."""

    embedding_matrix = (
        np.random.default_rng(123).normal(size=(50, 4)).astype(np.float32)
    )
    indices = np.arange(50, dtype=np.int64)
    first = cluster_centroid_indices(
        embedding_matrix=embedding_matrix, indices=indices, sample_limit=20, seed=9
    )
    second = cluster_centroid_indices(
        embedding_matrix=embedding_matrix, indices=indices, sample_limit=20, seed=9
    )
    assert len(first) == 20
    assert first == second


def test_hash_text_stability() -> None:
    """Ensure hash keys are stable for equal inputs."""

    key_one = hash_text(kind="steering", model="m", text="abc")
    key_two = hash_text(kind="steering", model="m", text="abc")
    key_three = hash_text(kind="steering", model="m", text="abcd")
    assert key_one == key_two
    assert key_one != key_three


def test_embed_batch_preserves_row_order() -> None:
    """Ensure batched embedding responses map back to input row order."""

    with tempfile.TemporaryDirectory(prefix="embed_batch_order_") as tmpdir:
        cache_path = Path(tmpdir) / "cache" / "embeddings.jsonl"
        fake_client = FakeClient()

        texts = ["math A", "code B", "misc C", "math A"]
        matrix, stats = asyncio.run(
            embed_texts_async(
                texts=texts,
                model_name="gemini-embedding-001",
                cache_path=cache_path,
                concurrency=3,
                batch_size=2,
                requests_per_minute=1000,
                client=fake_client,
                retry_config=RetryConfig(timeout_seconds=10, max_retries=1),
                progress_desc="embed-test",
            )
        )

        assert matrix.shape == (4, 4)
        assert np.allclose(matrix[0], matrix[3])
        assert not np.allclose(matrix[0], matrix[1])
        assert stats.hit_count == 0
        assert stats.miss_count == 3


def test_build_unique_text_index_maps_duplicate_rows() -> None:
    """Ensure duplicate strings map to a shared unique index."""

    unique_texts, row_to_unique_index = build_unique_text_index(
        texts=["alpha", "beta", "alpha", "alpha", "gamma"]
    )
    assert unique_texts == ["alpha", "beta", "gamma"]
    assert row_to_unique_index.tolist() == [0, 1, 0, 0, 2]


def test_cluster_unique_strings_maps_duplicate_labels_and_vectors() -> None:
    """Ensure unique-string clustering broadcasts labels/vectors back to duplicates."""

    with tempfile.TemporaryDirectory(prefix="unique_cluster_mapping_") as tmpdir:
        cache_path = Path(tmpdir) / "cache" / "embeddings.jsonl"
        fake_client = FakeClient()
        texts = ["math plan", "math plan", "code plan", "code plan", "other plan"]

        result = asyncio.run(
            cluster_texts_using_unique_strings(
                texts=texts,
                model_name="gemini-embedding-001",
                cache_path=cache_path,
                concurrency=3,
                batch_size=2,
                requests_per_minute=1000,
                min_cluster_size=2,
                max_cluster_size=2000,
                client=fake_client,
                retry_config=RetryConfig(timeout_seconds=10, max_retries=1),
                progress_desc="unique-cluster-test",
            )
        )

        assert result.unique_text_count == 3
        assert result.duplicate_row_count == 2
        assert result.row_labels.shape == (5,)
        assert result.row_embeddings.shape == (5, 4)
        assert int(result.row_labels[0]) == int(result.row_labels[1])
        assert int(result.row_labels[2]) == int(result.row_labels[3])
        assert np.allclose(result.row_embeddings[0], result.row_embeddings[1])
        assert np.allclose(result.row_embeddings[2], result.row_embeddings[3])


def test_assign_noise_fixed_centroids() -> None:
    """Verify noise assignment uses centroids and preserves non-noise labels."""

    with tempfile.TemporaryDirectory(prefix="cluster_noise_test_") as tmpdir:
        base = Path(tmpdir)
        clusters_path = base / "clusters_pass2.parquet"
        embeddings_path = base / "embeddings_pass2.npy"

        pass2_df = pd.DataFrame(
            {
                "section_id": ["a", "b", "c", "d", "e", "f"],
                "row_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
                "dataset_source": ["s"] * 6,
                "steering_text": ["x"] * 6,
                "cluster_pass1": [0, 0, 1, 1, 0, 1],
                "cluster_name_pass1": ["A", "A", "B", "B", "A", "B"],
                "cluster_pass2_raw": [0, 0, 1, 1, -1, -1],
            }
        )
        embeddings = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [5.0, 5.0],
                [5.1, 5.0],
                [0.05, 0.02],
                [4.95, 5.02],
            ],
            dtype=np.float32,
        )
        pass2_df.to_parquet(clusters_path, index=False)
        np.save(embeddings_path, embeddings)

        config = build_test_config(
            base_dir=base, transformed_path=base / "t.jsonl", og_path=base / "o.jsonl"
        )
        result = stage_assign_noise(config=config)
        final_df = pd.read_parquet(base / "clusters_final.parquet")

        assert result["noise_before"] == 2
        assert result["noise_after"] == 0
        assert int(final_df["noise_assigned"].sum()) == 2
        assert final_df.loc[~final_df["noise_assigned"], "cluster_pass2"].tolist()[
            :4
        ] == [0, 0, 1, 1]


def test_integration_dry_pipeline_with_mocked_client() -> None:
    """Run extract->tokens->cluster1->name1->cluster2->assign_noise->name2->report with mocked API."""

    with tempfile.TemporaryDirectory(prefix="cluster_pipeline_test_") as tmpdir:
        base = Path(tmpdir)
        output_dir = base / "analysis"
        transformed_path = base / "transformed.jsonl"
        og_path = base / "og.jsonl"

        transformed_rows = []
        og_rows = []
        for index in range(6):
            tag = "math" if index < 3 else "code"
            row_id = f"row-{index}"
            transformed_rows.append(
                {
                    "id": row_id,
                    "dataset_source": "unit-test",
                    "messages": [
                        {"role": "user", "content": f"question {index}"},
                        {
                            "role": "assistant",
                            "content": (
                                "<think>"
                                f"<steer>{tag} plan {index}</steer>"
                                f"<exec>{tag} execute {index}</exec>"
                                "</think>"
                            ),
                        },
                    ],
                    "transform_meta": {"source_batch": "fake-batch"},
                }
            )
            og_rows.append(
                {
                    "id": row_id,
                    "dataset_source": "unit-test",
                    "messages": [
                        {"role": "user", "content": f"question {index}"},
                        {
                            "role": "assistant",
                            "content": (
                                "<think>" f"old {tag} reasoning {index}" "</think>"
                            ),
                        },
                    ],
                }
            )

        write_jsonl(path=transformed_path, rows=transformed_rows)
        write_jsonl(path=og_path, rows=og_rows)

        config = build_test_config(
            base_dir=output_dir, transformed_path=transformed_path, og_path=og_path
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "cache").mkdir(parents=True, exist_ok=True)
        (output_dir / "plots").mkdir(parents=True, exist_ok=True)

        fake_client = FakeClient()
        extract_meta = stage_extract(config=config)
        token_meta = asyncio.run(stage_tokens_async(config=config, client=fake_client))
        cluster1_meta = asyncio.run(
            stage_cluster1_async(config=config, client=fake_client)
        )
        name1_meta = asyncio.run(stage_name1_async(config=config, client=fake_client))
        cluster2_meta = asyncio.run(
            stage_cluster2_async(config=config, client=fake_client)
        )
        noise_meta = stage_assign_noise(config=config)
        name2_meta = asyncio.run(stage_name2_async(config=config, client=fake_client))
        report_meta = stage_report(config=config)

        assert extract_meta["sections_emitted"] == 6
        assert token_meta["token_requests"] >= 12
        assert cluster1_meta["sections"] == 6
        assert name1_meta["clusters_named"] >= 1
        assert cluster2_meta["sections"] == 6
        assert noise_meta["rows"] == 6
        assert name2_meta["clusters_named"] >= 1
        assert Path(report_meta["report_md"]).exists()
        assert Path(report_meta["report_json"]).exists()

        final_df = pd.read_parquet(output_dir / "clusters_final.parquet")
        assert final_df.shape[0] == 6
        assert "cluster_name_final" in final_df.columns


def main() -> None:
    """Execute dry tests without pytest dependency."""

    test_parse_steering_execution_pairs_mixed_tags()
    test_cluster_centroid_indices_max_20_and_deterministic()
    test_hash_text_stability()
    test_embed_batch_preserves_row_order()
    test_build_unique_text_index_maps_duplicate_rows()
    test_cluster_unique_strings_maps_duplicate_labels_and_vectors()
    test_assign_noise_fixed_centroids()
    test_integration_dry_pipeline_with_mocked_client()
    print("ok: analyze steering clusters dry tests passed")


if __name__ == "__main__":
    main()
