from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
from dataclasses import dataclass
import io
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_sft_dataset import (
    PromptConfig,
    TransformConfig,
    load_seen_ids,
    stream_non_batch_transform_rows,
    run_async_transform_prompts,
)


@dataclass(frozen=True)
class FakeResponse:
    """Minimal async response with text payload.

    Args:
        text: Response text.
    """

    text: str


class FakeAsyncModels:
    """Mock async Gemini models API with concurrency tracking."""

    def __init__(self) -> None:
        self.inflight = 0
        self.max_inflight = 0

    async def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: object | None = None,
    ) -> FakeResponse:
        """Return deterministic output after a short async delay.

        Args:
            model: Model identifier.
            contents: Prompt text.
            config: Optional generation config.

        Returns:
            Response object carrying deterministic text.
        """

        del model, config
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0.01)
        self.inflight -= 1
        return FakeResponse(text=f"done:{contents}")


class FailingAsyncModels(FakeAsyncModels):
    """Mock async models API that fails for one prompt."""

    async def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: object | None = None,
    ) -> FakeResponse:
        """Raise one error to test graceful skip behavior.

        Args:
            model: Model identifier.
            contents: Prompt text.
            config: Optional generation config.

        Returns:
            Fake response object.
        """

        if contents == "b":
            raise RuntimeError("500 INTERNAL")
        return await super().generate_content(
            model=model,
            contents=contents,
            config=config,
        )


class FakeAio:
    """Mock aio wrapper exposing async models."""

    def __init__(self) -> None:
        self.models = FakeAsyncModels()


class FakeClient:
    """Mock client exposing `aio.models.generate_content`."""

    def __init__(self) -> None:
        self.aio = FakeAio()


class FailingClient:
    """Mock client with one forced async failure."""

    def __init__(self) -> None:
        self.aio = FakeAio()
        self.aio.models = FailingAsyncModels()


def build_test_config() -> TransformConfig:
    """Build transform config for async transform tests.

    Returns:
        Transform config with small concurrency limit.
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
        batch=False,
        max_concurrent_requests=2,
        retry_limit=2,
        retry_sleep_seconds=0.0,
        batch_poll_seconds=0.0,
        max_rows=4,
        dry_run=False,
    )


def main() -> None:
    """Verify async non-batch transform streams resumable rows only.

    Example:
        uv run python BuildSFTDataset/tests/test_transform_async.py
    """

    fake_client = FakeClient()
    result = asyncio.run(
        run_async_transform_prompts(
            client=fake_client,
            model=build_test_config(),
            system_prompt="system",
            user_prompts=["a", "b", "c", "d"],
            original_blocks=["orig-a", "orig-b", "orig-c", "orig-d"],
        )
    )

    assert result.outputs == ["done:a", "done:b", "done:c", "done:d"]
    assert result.failed_indexes == []
    assert fake_client.aio.models.max_inflight <= 2

    failing_client = FailingClient()
    failing_stdout = io.StringIO()
    with redirect_stdout(failing_stdout):
        failing_result = asyncio.run(
            run_async_transform_prompts(
                client=failing_client,
                model=build_test_config(),
                system_prompt="system",
                user_prompts=["a", "b", "c"],
                original_blocks=["orig-a", "orig-b", "orig-c"],
            )
        )
    assert failing_result.outputs == ["done:a", "", "done:c"]
    assert failing_result.failed_indexes == [1]
    assert "500 INTERNAL" in failing_result.failed_errors[1]
    assert "request_index=1" in failing_stdout.getvalue()

    with tempfile.TemporaryDirectory(prefix="buildsft_async_stream_") as tmpdir:
        output_path = Path(tmpdir) / "transformed_output.jsonl"
        rows = [
            {
                "id": "row-1",
                "dataset_source": "math",
                "messages": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "<think>a</think>\nanswer"},
                ],
            },
            {
                "id": "row-2",
                "dataset_source": "math",
                "messages": [
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "<think>b</think>\nanswer"},
                ],
            },
        ]
        rows_emitted, failed_rows, failed_blocks, _ = asyncio.run(
            stream_non_batch_transform_rows(
                client=FakeClient(),
                model=build_test_config(),
                prompts=PromptConfig(
                    system_prompt_path=Path("system_prompt.md"),
                    user_prompt_path=Path("user_prompt.md"),
                ),
                system_prompt="system",
                user_template="{think_text}",
                rows_to_transform=rows,
                output_path=output_path,
            )
        )
        assert rows_emitted == 2
        assert failed_rows == 0
        assert failed_blocks == 0
        assert load_seen_ids(path=output_path) == {"row-1", "row-2"}

    with tempfile.TemporaryDirectory(prefix="buildsft_async_stream_fail_") as tmpdir:
        output_path = Path(tmpdir) / "transformed_output.jsonl"
        rows = [
            {
                "id": "row-1",
                "dataset_source": "math",
                "messages": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "<think>a</think>\nanswer"},
                ],
            },
            {
                "id": "row-2",
                "dataset_source": "math",
                "messages": [
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "<think>b</think>\nanswer"},
                ],
            },
        ]
        failing_stdout = io.StringIO()
        with redirect_stdout(failing_stdout):
            rows_emitted, failed_rows, failed_blocks, failed_errors = asyncio.run(
                stream_non_batch_transform_rows(
                    client=FailingClient(),
                    model=build_test_config(),
                    prompts=PromptConfig(
                        system_prompt_path=Path("system_prompt.md"),
                        user_prompt_path=Path("user_prompt.md"),
                    ),
                    system_prompt="system",
                    user_template="{think_text}",
                    rows_to_transform=rows,
                    output_path=output_path,
                )
            )
        assert rows_emitted == 1
        assert failed_rows == 1
        assert failed_blocks == 1
        assert failed_errors == ["500 INTERNAL"]
        assert "row_id=row-2" in failing_stdout.getvalue()
        assert load_seen_ids(path=output_path) == {"row-1"}
    print("ok: async transform test passed")


if __name__ == "__main__":
    main()
