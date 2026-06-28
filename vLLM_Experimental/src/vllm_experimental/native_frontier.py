"""Native branch-product frontier data objects."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NativeFrontierItem:
    """One visible trajectory prefix that may still branch."""

    root_prompt: str
    root_prompt_token_ids: list[int]
    completion_token_ids: list[int]
    request_index: int
    branch_path: tuple[int, ...] = ()
    depth: int = 0

    def prompt_input(self) -> str | dict[str, list[int]]:
        """Return a vLLM prompt input for this frontier item."""

        if not self.completion_token_ids:
            return self.root_prompt
        return {
            "prompt_token_ids": self.root_prompt_token_ids + self.completion_token_ids
        }

    def prefix_output_token_ids(
        self, *, assistant_prefill_token_ids: list[int]
    ) -> list[int]:
        """Return grammar replay tokens already present in the prompt."""

        return assistant_prefill_token_ids + self.completion_token_ids

    def remaining_tokens(self, *, max_tokens: int) -> int:
        """Return generation budget remaining for this trajectory."""

        return max(0, max_tokens - len(self.completion_token_ids))

    def child(
        self,
        *,
        candidate_id: int,
        token_ids: list[int],
    ) -> "NativeFrontierItem":
        """Return the next-depth frontier item for one branch product."""

        return NativeFrontierItem(
            root_prompt=self.root_prompt,
            root_prompt_token_ids=self.root_prompt_token_ids,
            completion_token_ids=self.completion_token_ids + token_ids,
            request_index=self.request_index,
            branch_path=self.branch_path + (candidate_id,),
            depth=self.depth + 1,
        )


@dataclass(frozen=True)
class NativeLeafOutput:
    """One completed visible native branch product."""

    request_index: int
    branch_path: tuple[int, ...]
    depth: int
    token_ids: list[int]
    text: str
    finish_reason: str

    def sample_payload(
        self,
        *,
        row: dict[str, object],
        prompt: str,
    ) -> dict[str, Any]:
        """Return JSON payload for compact trajectory samples."""

        return {
            "mode": row["mode"],
            "prompt_concurrency": row["prompt_concurrency"],
            "request_prompt_batch_size": row["request_prompt_batch_size"],
            "request_index": self.request_index,
            "choice_index": self.branch_path[-1] if self.branch_path else 0,
            "branch_path": list(self.branch_path),
            "branch_depth": self.depth,
            "prompt": prompt,
            "prompt_char_count": len(prompt),
            "prompt_tail": prompt[-1200:],
            "text": self.text,
            "token_count": len(self.token_ids),
            "token_ids": [int(token_id) for token_id in self.token_ids],
            "finish_reason": self.finish_reason,
        }


def candidate_id_from_choice(*, choice: Any, fallback: int) -> int:
    """Read vLLM candidate ids from branch-product choices."""

    return int(getattr(choice, "index", fallback))


def chunked_frontier(
    *, items: list[NativeFrontierItem], size: int
) -> list[list[NativeFrontierItem]]:
    """Split frontier items into non-empty chunks."""

    assert size >= 1, "frontier chunk size must be positive"
    return [items[index : index + size] for index in range(0, len(items), size)]


def tokenized_text(*, tokenizer: Any, text: str) -> list[int]:
    """Return token ids without adding special tokens."""

    token_ids = tokenizer.encode(text=text, add_special_tokens=False)
    assert token_ids, f"text tokenized empty: {text!r}"
    return [int(token_id) for token_id in token_ids]


def native_frontier_items(
    *,
    prompts: list[str],
    tokenizer: Any,
    request_offset: int,
) -> list[NativeFrontierItem]:
    """Create initial recursive native frontier items."""

    return [
        NativeFrontierItem(
            root_prompt=prompt,
            root_prompt_token_ids=tokenized_text(tokenizer=tokenizer, text=prompt),
            completion_token_ids=[],
            request_index=request_offset + local_index,
        )
        for local_index, prompt in enumerate(prompts)
    ]


def native_sampling_params(
    *,
    vllm_mod: Any,
    item: NativeFrontierItem,
    tree_search: dict[str, object],
    assistant_prefill_token_ids: list[int],
    max_tokens: int,
) -> Any:
    """Return SamplingParams for one recursive native frontier item."""

    payload = dict(tree_search)
    payload["prefix_output_token_ids"] = item.prefix_output_token_ids(
        assistant_prefill_token_ids=assistant_prefill_token_ids
    )
    payload["branch_depth_start"] = item.depth
    return vllm_mod.SamplingParams(
        max_tokens=item.remaining_tokens(max_tokens=max_tokens),
        temperature=1.0,
        extra_args={"vllm_experimental": payload},
    )


def decode_token_ids(*, tokenizer: Any, token_ids: list[int]) -> str:
    """Decode a token-id list with a vLLM/HF tokenizer."""

    return str(tokenizer.decode(token_ids))


def choice_token_ids(*, choice: Any) -> list[int]:
    """Return generated token ids from a vLLM completion choice."""

    return [int(token_id) for token_id in choice.token_ids]


def run_native_frontier_batch(
    *,
    llm: Any,
    vllm_mod: Any,
    items: list[NativeFrontierItem],
    tree_search: dict[str, object],
    assistant_prefill_token_ids: list[int],
    max_tokens: int,
) -> list[Any]:
    """Generate one batch of recursive native frontier items."""

    sampling_params = [
        native_sampling_params(
            vllm_mod=vllm_mod,
            item=item,
            tree_search=tree_search,
            assistant_prefill_token_ids=assistant_prefill_token_ids,
            max_tokens=max_tokens,
        )
        for item in items
    ]
    return llm.generate(
        prompts=[item.prompt_input() for item in items],
        sampling_params=sampling_params,
        use_tqdm=False,
    )


def write_frontier_trace(*, path: Path, payload: dict[str, object]) -> None:
    """Append one compact recursive frontier event."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def frontier_batch_payload(
    *,
    event: str,
    items: list[NativeFrontierItem],
    pending_after_pop: int,
) -> dict[str, object]:
    """Return one compact frontier-batch trace payload."""

    depths = [item.depth for item in items]
    return {
        "event": event,
        "time_unix": time.time(),
        "batch_item_count": len(items),
        "pending_after_pop": pending_after_pop,
        "request_indexes": [item.request_index for item in items],
        "depth_min": min(depths),
        "depth_max": max(depths),
        "branch_paths": [list(item.branch_path) for item in items],
    }


def handle_native_frontier_output(
    *,
    item: NativeFrontierItem,
    request_output: Any,
    tokenizer: Any,
    max_tokens: int,
    branch_depth: int,
    trace_path: Path,
) -> tuple[list[NativeFrontierItem], list[NativeLeafOutput]]:
    """Split one vLLM output into next frontier items and leaves."""

    next_items: list[NativeFrontierItem] = []
    leaves: list[NativeLeafOutput] = []
    choices = list(request_output.outputs)
    for choice_index, choice in enumerate(choices):
        token_ids = choice_token_ids(choice=choice)
        candidate_id = candidate_id_from_choice(
            choice=choice,
            fallback=choice_index,
        )
        full_token_ids = item.completion_token_ids + token_ids
        has_branch_product = len(choices) > 1
        child = item.child(candidate_id=candidate_id, token_ids=token_ids)
        can_expand = (
            has_branch_product
            and child.depth < branch_depth
            and len(full_token_ids) < max_tokens
        )
        if can_expand:
            next_items.append(child)
        else:
            leaves.append(
                NativeLeafOutput(
                    request_index=item.request_index,
                    branch_path=(
                        child.branch_path if has_branch_product else item.branch_path
                    ),
                    depth=child.depth if has_branch_product else item.depth,
                    token_ids=full_token_ids,
                    text=decode_token_ids(
                        tokenizer=tokenizer,
                        token_ids=full_token_ids,
                    ),
                    finish_reason=str(getattr(choice, "finish_reason", "")),
                )
            )
        write_frontier_trace(
            path=trace_path,
            payload={
                "event": "frontier_choice",
                "request_index": item.request_index,
                "parent_branch_path": list(item.branch_path),
                "branch_path": list(child.branch_path),
                "parent_depth": item.depth,
                "branch_depth": child.depth,
                "candidate_id": candidate_id,
                "choice_count": len(choices),
                "choice_token_count": len(token_ids),
                "cumulative_token_count": len(full_token_ids),
                "expanded": can_expand,
            },
        )
    return next_items, leaves


def run_native_frontier_chunk(
    *,
    llm: Any,
    vllm_mod: Any,
    tokenizer: Any,
    prompts: list[str],
    tree_search: dict[str, object],
    max_tokens: int,
    request_offset: int,
    trace_path: Path,
    frontier_batch_size: int,
    assistant_prefill: str,
) -> list[NativeLeafOutput]:
    """Run recursive native branch products to the configured depth."""

    raw_branch_depth = tree_search.get("branch_depth", 4)
    assert isinstance(raw_branch_depth, (int, str)), "branch_depth must be scalar"
    branch_depth = int(raw_branch_depth)
    assert branch_depth >= 1, "branch_depth must be positive"
    assistant_ids = tokenized_text(tokenizer=tokenizer, text=assistant_prefill)
    frontier = native_frontier_items(
        prompts=prompts,
        tokenizer=tokenizer,
        request_offset=request_offset,
    )
    leaves: list[NativeLeafOutput] = []
    while frontier:
        current, frontier = frontier, []
        for item_batch in chunked_frontier(items=current, size=frontier_batch_size):
            write_frontier_trace(
                path=trace_path,
                payload=frontier_batch_payload(
                    event="frontier_batch_start",
                    items=item_batch,
                    pending_after_pop=len(frontier),
                ),
            )
            outputs = run_native_frontier_batch(
                llm=llm,
                vllm_mod=vllm_mod,
                items=item_batch,
                tree_search=tree_search,
                assistant_prefill_token_ids=assistant_ids,
                max_tokens=max_tokens,
            )
            assert len(item_batch) == len(outputs), "frontier output count mismatch"
            write_frontier_trace(
                path=trace_path,
                payload=frontier_batch_payload(
                    event="frontier_batch_complete",
                    items=item_batch,
                    pending_after_pop=len(frontier),
                ),
            )
            for item, request_output in zip(item_batch, outputs):
                next_items, next_leaves = handle_native_frontier_output(
                    item=item,
                    request_output=request_output,
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    branch_depth=branch_depth,
                    trace_path=trace_path,
                )
                frontier.extend(next_items)
                leaves.extend(next_leaves)
    return leaves
