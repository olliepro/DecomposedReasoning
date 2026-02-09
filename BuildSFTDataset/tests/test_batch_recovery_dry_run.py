from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_sft_dataset import TransformConfig, run_existing_batch_requests


@dataclass(frozen=True)
class FakeState:
    """Minimal state object with a `name` field."""

    name: str


@dataclass(frozen=True)
class FakeResponse:
    """Minimal model response object with direct text."""

    text: str


@dataclass(frozen=True)
class FakeInlinedResponse:
    """Minimal inlined batch response item."""

    metadata: dict[str, str]
    response: object | None = None
    error: object | None = None


@dataclass(frozen=True)
class FakeDestination:
    """Batch destination wrapper for inlined responses."""

    inlined_responses: list[FakeInlinedResponse]
    file_name: str | None = None


@dataclass(frozen=True)
class FakeBatchJob:
    """Minimal batch job for recovery tests."""

    name: str
    state: FakeState
    dest: FakeDestination


class FakeModels:
    """Fake models endpoint to test retry fallback calls."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_content(self, *, model: str, contents: object, config: object) -> FakeResponse:
        """Return deterministic fallback responses by call order.

        Args:
            model: Model identifier.
            contents: Request payload.
            config: Generation config.

        Returns:
            Fake response object with text.
        """
        del model, contents, config
        self.calls += 1
        return FakeResponse(text=f"fallback-{self.calls}")


class FakeBatches:
    """Fake batches endpoint with a single job."""

    def __init__(self, job: FakeBatchJob) -> None:
        self.job = job

    def get(self, *, name: str) -> FakeBatchJob:
        """Return job regardless of lookup name.

        Args:
            name: Batch name.

        Returns:
            Stored fake batch job.
        """
        del name
        return self.job


class FakeFiles:
    """Unused files endpoint placeholder."""

    def download(self, *, file: str, config: object | None = None) -> bytes:
        """Fail if file download path is reached in this test."""
        del file, config
        raise AssertionError("Did not expect file download path in dry test")


class FakeClient:
    """Minimal fake GenAI client for batch recovery."""

    def __init__(self, job: FakeBatchJob) -> None:
        self.batches = FakeBatches(job=job)
        self.models = FakeModels()
        self.files = FakeFiles()


def build_test_model() -> TransformConfig:
    """Build transform config for dry tests.

    Returns:
        Transform config with tiny retry settings.
    """
    return TransformConfig(
        mode="gemini",
        model_id="gemini-3-flash-preview",
        api_key="DUMMY",
        project_id=None,
        location=None,
        max_output_tokens=20000,
        temperature=0.0,
        thinking_level="LOW",
        batch=True,
        retry_limit=2,
        retry_sleep_seconds=0.0,
        batch_poll_seconds=0.0,
        max_rows=2,
        dry_run=False,
    )


def main() -> None:
    """Validate that recovery maps outputs by request index and retries failures."""
    responses = [
        FakeInlinedResponse(metadata={"request_index": "2"}, response=FakeResponse(text="third")),
        FakeInlinedResponse(metadata={"request_index": "0"}, response=FakeResponse(text="first")),
        FakeInlinedResponse(metadata={"request_index": "1"}, error={"code": 4, "message": "timeout"}),
    ]
    fake_job = FakeBatchJob(
        name="batches/abc123",
        state=FakeState(name="JOB_STATE_SUCCEEDED"),
        dest=FakeDestination(inlined_responses=responses),
    )
    fake_client = FakeClient(job=fake_job)
    model = build_test_model()
    inline_requests = [
        {"contents": [{"parts": [{"text": "req-0"}]}], "config": {}},
        {"contents": [{"parts": [{"text": "req-1"}]}], "config": {}},
        {"contents": [{"parts": [{"text": "req-2"}]}], "config": {}},
    ]

    result = run_existing_batch_requests(
        client=fake_client,  # type: ignore[arg-type]
        model=model,
        inline_requests=inline_requests,
        batch_job_name="abc123",
    )

    assert result.batch_job_name == "batches/abc123"
    assert result.outputs == ["first", "fallback-1", "third"]
    assert fake_client.models.calls == 1
    print("ok: batch recovery dry-run test passed")


if __name__ == "__main__":
    main()
