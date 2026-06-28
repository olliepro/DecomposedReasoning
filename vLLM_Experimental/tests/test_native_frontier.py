from __future__ import annotations

from pathlib import Path
from typing import Any

from vllm_experimental.run_benchmark_job import run_native_frontier_chunk


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return ",".join(str(token_id) for token_id in token_ids)


class FakeSamplingParams:
    def __init__(
        self,
        *,
        max_tokens: int,
        temperature: float,
        extra_args: dict[str, object],
    ) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_args = extra_args


class FakeVllmModule:
    SamplingParams = FakeSamplingParams


class FakeChoice:
    def __init__(self, *, index: int, token_ids: list[int]) -> None:
        self.index = index
        self.token_ids = token_ids
        self.finish_reason = "stop"


class FakeRequestOutput:
    def __init__(self, *, outputs: list[FakeChoice]) -> None:
        self.outputs = outputs


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        prompts: list[str | dict[str, list[int]]],
        sampling_params: list[FakeSamplingParams],
        use_tqdm: bool,
    ) -> list[FakeRequestOutput]:
        assert not use_tqdm
        depths = [
            int(
                param.extra_args["vllm_experimental"]["branch_depth_start"]  # type: ignore[index]
            )
            for param in sampling_params
        ]
        self.calls.append({"prompts": prompts, "depths": depths})
        return [
            FakeRequestOutput(
                outputs=[
                    FakeChoice(index=0, token_ids=[10 + depth, 0]),
                    FakeChoice(index=1, token_ids=[10 + depth, 1]),
                ]
            )
            for depth in depths
        ]


def test_native_frontier_recurses_to_branch_depth(tmp_path: Path) -> None:
    llm = FakeLLM()

    leaves = run_native_frontier_chunk(
        llm=llm,
        vllm_mod=FakeVllmModule,
        tokenizer=FakeTokenizer(),
        prompts=["prompt"],
        tree_search={"branch_depth": 3},
        max_tokens=64,
        request_offset=0,
        trace_path=tmp_path / "frontier.jsonl",
        frontier_batch_size=128,
        assistant_prefill="<think>\n",
    )

    assert len(leaves) == 8
    assert {len(leaf.branch_path) for leaf in leaves} == {3}
    assert {leaf.depth for leaf in leaves} == {3}
    assert [call["depths"][0] for call in llm.calls] == [0, 1, 2]
    assert isinstance(llm.calls[0]["prompts"][0], str)
    assert isinstance(llm.calls[1]["prompts"][0], dict)
    trace_text = (tmp_path / "frontier.jsonl").read_text(encoding="utf-8")
    assert trace_text.count("frontier_choice") == 14
    assert trace_text.count("frontier_batch_start") == 3
    assert trace_text.count("frontier_batch_complete") == 3
