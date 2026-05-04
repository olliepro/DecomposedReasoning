from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import httpx
from pydantic import BaseModel, Field, ValidationError

from .prompt_routing import choose_prompt_path as resolve_prompt_path

TRACE_TAG_RE = re.compile(r"<(steer|exec)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class TraceBlock:
    type: Literal["steer", "exec"]
    text: str


def trace_block_to_dict(block: TraceBlock) -> Dict[str, str]:
    """Convert a `TraceBlock` into the serialized block payload format."""
    return {"type": block.type, "text": block.text}


def normalize_trace_block(block: TraceBlock | Mapping[str, Any]) -> TraceBlock:
    """Convert a trace block-like mapping into a TraceBlock.

    Inputs:
        block: A `TraceBlock` or a mapping with `type` and `text` keys.

    Outputs:
        A validated `TraceBlock` instance.
    """
    if isinstance(block, TraceBlock):
        return block
    return TraceBlock(type=block["type"], text=block["text"])


class OutputBlock(BaseModel):
    type: Literal["steer", "exec"]
    text: str = Field(min_length=1)


class InterventionWindow(BaseModel):
    blocks: List[OutputBlock] = Field(min_length=2)


class BridgeJudgeResult(BaseModel):
    decision: Literal["keep_suffix", "regenerate_suffix"]
    reason: str = Field(min_length=1)


class TokenCounter:
    """
    Token counter with three fallback modes:
    1) Hugging Face tokenizer if a tokenizer name/path is provided.
    2) tiktoken cl100k_base.
    3) crude whitespace fallback.

    Exact enforcement depends on tokenizer choice. For production, prefer
    the tokenizer of the model you care about most.
    """

    def __init__(self, tokenizer_name: Optional[str] = None):
        self.method = "whitespace_approx"
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None

        if tokenizer_name:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name, trust_remote_code=True
                )
                self.method = "hf"
                return
            except Exception:
                pass

        try:
            import tiktoken

            self._tokenizer = tiktoken.get_encoding("cl100k_base")
            self.method = "tiktoken_cl100k"
        except Exception:
            self._tokenizer = None
            self.method = "whitespace_approx"

    def count(self, text: str) -> int:
        if self.method == "hf":
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        if self.method == "tiktoken_cl100k":
            return len(self._tokenizer.encode(text))
        # crude approximation
        return max(1, len(re.findall(r"\S+", text)))


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
    return rows


def dump_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_trace_text(record: Dict[str, Any]) -> str:
    """Extract interleaved trace text from a supported record shape.

    Inputs:
        record: Dataset row containing trace text directly or inside chat messages.

    Outputs:
        The assistant reasoning trace as a string.

    Example:
        >>> extract_trace_text({"assistant": "<think><steer>a</steer><exec>b</exec></think>"})
        '<steer>a</steer><exec>b</exec>'
    """
    if "trace" in record and isinstance(record["trace"], str):
        return record["trace"]
    if "trace_blocks" in record and isinstance(record["trace_blocks"], list):
        return render_trace(
            [
                TraceBlock(type=b["type"], text=b["text"])
                for b in record["trace_blocks"]
            ],
            wrap_think=False,
        )
    if "assistant" in record and isinstance(record["assistant"], str):
        m = THINK_RE.search(record["assistant"])
        if m:
            return m.group(1)
        return record["assistant"]
    messages = record.get("messages")
    if isinstance(messages, list):
        assistant_messages = [
            item
            for item in messages
            if isinstance(item, dict)
            and item.get("role") == "assistant"
            and isinstance(item.get("content"), str)
        ]
        if assistant_messages:
            assistant_text = assistant_messages[-1]["content"]
            m = THINK_RE.search(assistant_text)
            if m:
                return m.group(1)
            return assistant_text
    raise KeyError(
        "Record must contain one of: trace, trace_blocks, assistant, messages[role=assistant]"
    )


def parse_trace_text(trace_text: str) -> List[TraceBlock]:
    blocks: List[TraceBlock] = []
    for match in TRACE_TAG_RE.finditer(trace_text):
        block_type = match.group(1).lower()
        text = match.group(2).strip()
        if text:
            blocks.append(TraceBlock(type=block_type, text=text))
    return blocks


def render_trace(blocks: Sequence[TraceBlock], wrap_think: bool = False) -> str:
    body = "\n\n".join(f"<{b.type}>{b.text}</{b.type}>" for b in blocks)
    if wrap_think:
        return f"<think>\n{body}\n</think>"
    return body


def render_open_exec_trace(blocks: Sequence[TraceBlock]) -> str:
    """Render blocks so continuation resumes before the final `</exec>`.

    Inputs:
        blocks: Interleaved trace blocks already accepted into the augmented prefix.

    Outputs:
        Assistant prefill text ending inside `<think>` and inside the final `<exec>`.

    Example:
        >>> render_open_exec_trace(
        ...     [TraceBlock(type="steer", text="plan"), TraceBlock(type="exec", text="work")]
        ... )
        '<think>\\n<steer>plan</steer>\\n\\n<exec>work'
    """
    normalized_blocks = list(blocks)
    assert normalized_blocks, "regen seed requires at least one block"
    assert normalized_blocks[-1].type == "exec", "regen seed must end on exec"
    closed_prefix = normalized_blocks[:-1]
    prefix_text = "\n\n".join(f"<{b.type}>{b.text}</{b.type}>" for b in closed_prefix)
    final_exec = normalized_blocks[-1]
    open_exec = f"<exec>{final_exec.text}"
    body_parts = [part for part in [prefix_text, open_exec] if part]
    body = "\n\n".join(body_parts)
    return f"<think>\n{body}"


def render_window(blocks: Sequence[TraceBlock | Mapping[str, Any]]) -> str:
    normalized = [normalize_trace_block(block) for block in blocks]
    return "\n\n".join(f"<{b.type}>{b.text}</{b.type}>" for b in normalized)


def ensure_interleaved(
    blocks: Sequence[TraceBlock],
    *,
    token_counter: TokenCounter,
    exec_token_limit: int,
) -> List[str]:
    errors: List[str] = []
    if not blocks:
        return ["no steer/exec blocks found"]

    if len(blocks) % 2 != 0:
        errors.append(f"block count must be even, got {len(blocks)}")

    if blocks[0].type != "steer":
        errors.append(f"first block must be steer, got {blocks[0].type}")

    for i, block in enumerate(blocks):
        expected = "steer" if i % 2 == 0 else "exec"
        if block.type != expected:
            errors.append(f"block index {i} must be {expected}, got {block.type}")
        if not block.text.strip():
            errors.append(f"block index {i} is empty")
        if block.type == "exec":
            n_tokens = token_counter.count(block.text)
            if n_tokens >= exec_token_limit:
                errors.append(
                    f"exec block index {i} has {n_tokens} tokens, exceeds limit < {exec_token_limit}"
                )
    return errors


def pair_count(blocks: Sequence[TraceBlock]) -> int:
    return len(blocks) // 2


def slice_pairs(blocks: Sequence[TraceBlock], num_pairs: int) -> List[TraceBlock]:
    return list(blocks[: 2 * num_pairs])


def get_pair(
    blocks: Sequence[TraceBlock], pair_idx: int
) -> Tuple[TraceBlock, TraceBlock]:
    a = blocks[2 * pair_idx]
    b = blocks[2 * pair_idx + 1]
    return a, b


def build_style_window(
    prefix_blocks: Sequence[TraceBlock], style_window_pairs: int = 2
) -> str:
    n_pairs = pair_count(prefix_blocks)
    start_pair = max(0, n_pairs - style_window_pairs)
    return render_window(prefix_blocks[2 * start_pair :])


def get_next_original_context(
    all_blocks: Sequence[TraceBlock], cut_after_pairs: int
) -> Tuple[str, str]:
    total_pairs = pair_count(all_blocks)
    if cut_after_pairs >= total_pairs:
        return "", ""
    steer_block, exec_block = get_pair(all_blocks, cut_after_pairs)
    exec_preview = summarize_preview(exec_block.text)
    return steer_block.text, exec_preview


def summarize_preview(text: str, max_words: int = 30) -> str:
    words = re.findall(r"\S+", text)
    if not words:
        return ""
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]) + " ..."


def select_cut_pair_index(
    total_pairs: int,
    preferred_slots: Sequence[str],
    rng: random.Random,
) -> int:
    """
    Returns number of pairs kept in the prefix.
    Must be in [1, total_pairs - 1] so that there is a non-empty suffix.
    """
    if total_pairs < 2:
        raise ValueError("Need at least 2 pairs to insert before a non-empty suffix.")

    candidates = list(range(1, total_pairs))
    if not preferred_slots:
        return rng.choice(candidates)

    # Partition candidates by early/mid/late on pair boundaries.
    early_end = max(1, math.floor((total_pairs - 1) * 0.25))
    late_start = min(total_pairs - 1, max(1, math.ceil((total_pairs - 1) * 0.75)))
    buckets = {
        "early": [i for i in candidates if i <= early_end],
        "mid": [i for i in candidates if early_end < i < late_start],
        "late": [i for i in candidates if i >= late_start],
    }

    preferred_candidates: List[int] = []
    for slot in preferred_slots:
        preferred_candidates.extend(buckets.get(slot, []))
    preferred_candidates = sorted(set(preferred_candidates))

    if preferred_candidates:
        return rng.choice(preferred_candidates)
    return rng.choice(candidates)


def slice_pair_blocks(
    blocks: Sequence[TraceBlock], start_pair: int, num_pairs: int = 1
) -> List[TraceBlock]:
    """Return a contiguous slice of whole pairs from an interleaved block sequence."""
    start = start_pair * 2
    end = start + num_pairs * 2
    return list(blocks[start:end])


def load_template(path: str | Path) -> Template:
    return Template(Path(path).read_text(encoding="utf-8"))


def render_prompt(path: str | Path, values: Dict[str, Any]) -> str:
    template = load_template(path)
    normalized = {k: ("" if v is None else v) for k, v in values.items()}
    return template.safe_substitute(normalized)


def build_intervention_schema(exact_pairs: int) -> Dict[str, Any]:
    exact_blocks = exact_pairs * 2
    return {
        "name": "intervention_window",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "blocks": {
                    "type": "array",
                    "minItems": exact_blocks,
                    "maxItems": exact_blocks,
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["steer", "exec"]},
                            "text": {"type": "string", "minLength": 1},
                        },
                        "required": ["type", "text"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["blocks"],
            "additionalProperties": False,
        },
    }


def build_bridge_judge_schema() -> Dict[str, Any]:
    return {
        "name": "bridge_judge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["keep_suffix", "regenerate_suffix"],
                },
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["decision", "reason"],
            "additionalProperties": False,
        },
    }


class OpenRouterAsyncClient:
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-20b",
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        timeout_s: int = 120,
        use_response_healing: bool = False,
        provider_require_parameters: bool = True,
        provider_data_collection: Optional[str] = "deny",
        reasoning_effort: Optional[str] = "low",
        temperature: float = 0.4,
        max_tokens: int = 1200,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.site_url = site_url
        self.site_name = site_name
        self.timeout_s = timeout_s
        self.use_response_healing = use_response_healing
        self.provider_require_parameters = provider_require_parameters
        self.provider_data_collection = provider_data_collection
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self._client: Optional[httpx.AsyncClient] = None

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-OpenRouter-Title"] = self.site_name
        return headers

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=self._headers(),
                timeout=httpx.Timeout(self.timeout_s),
                limits=httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_keepalive_connections,
                ),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "OpenRouterAsyncClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": json_schema,
            },
            "provider": {},
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        if self.provider_require_parameters:
            body["provider"]["require_parameters"] = True
        if self.provider_data_collection is not None:
            body["provider"]["data_collection"] = self.provider_data_collection
        if not body["provider"]:
            body.pop("provider")

        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort

        if self.use_response_healing:
            body["plugins"] = [{"id": "response-healing"}]

        client = await self._ensure_client()
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=body,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(
                f"Unexpected OpenRouter response shape: {json.dumps(payload)[:1200]}"
            ) from e

        return parse_openrouter_json_content(content)


# Backward-compatible name for existing imports.
OpenRouterClient = OpenRouterAsyncClient


def parse_openrouter_json_content(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON content from OpenRouter: {content}")
    if isinstance(content, list):
        # Some providers may return a content parts array.
        text_parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        joined = "".join(text_parts).strip()
        return json.loads(joined)
    raise TypeError(f"Unsupported content type from OpenRouter: {type(content)!r}")


def validate_generated_window(
    obj: Dict[str, Any],
    *,
    requested_pairs: int,
    required_first_steer: str,
    enforce_first_steer_exact: bool = True,
    token_counter: TokenCounter,
    exec_token_limit: int,
) -> Tuple[List[TraceBlock], List[str]]:
    errors: List[str] = []
    try:
        parsed = InterventionWindow.model_validate(obj)
    except ValidationError as e:
        return [], [f"structured output failed pydantic validation: {e}"]

    blocks = [TraceBlock(type=b.type, text=b.text.strip()) for b in parsed.blocks]

    if len(blocks) != requested_pairs * 2:
        errors.append(f"expected {requested_pairs * 2} blocks, got {len(blocks)}")

    interleave_errors = ensure_interleaved(
        blocks,
        token_counter=token_counter,
        exec_token_limit=exec_token_limit,
    )
    errors.extend(interleave_errors)

    if blocks:
        if blocks[0].type != "steer":
            errors.append("first generated block is not steer")
        elif (
            enforce_first_steer_exact
            and blocks[0].text.strip() != required_first_steer.strip()
        ):
            errors.append(
                f'first steer must equal variant exactly: "{required_first_steer}" vs "{blocks[0].text}"'
            )

    return blocks, errors


def choose_mode(intervention_spec: Dict[str, Any], rng: random.Random) -> str:
    weights = intervention_spec.get("mode_weight_hint") or {}
    items = [(k, v) for k, v in weights.items() if v > 0]
    if not items:
        return intervention_spec["recommended_editor_mode"]
    modes, probs = zip(*items)
    total = sum(probs)
    x = rng.random() * total
    acc = 0.0
    for mode, p in zip(modes, probs):
        acc += p
        if x <= acc:
            return mode
    return modes[-1]


def choose_variant(intervention_spec: Dict[str, Any], rng: random.Random) -> str:
    return rng.choice(intervention_spec["variants"])


def choose_pairs_to_generate(
    intervention_spec: Dict[str, Any], rng: random.Random
) -> int:
    cfg = intervention_spec["pairs_to_generate"]
    mn, mx, default = int(cfg["min"]), int(cfg["max"]), int(cfg["default"])
    if mn == mx:
        return mn
    # Favor the default while still exploring the allowed range.
    candidates = list(range(mn, mx + 1))
    weights = [3.0 if x == default else 1.0 for x in candidates]
    total = sum(weights)
    x = rng.random() * total
    acc = 0.0
    for c, w in zip(candidates, weights):
        acc += w
        if x <= acc:
            return c
    return default


def build_prompt_values(
    *,
    record: Dict[str, Any],
    all_blocks: Sequence[TraceBlock],
    prefix_blocks: Sequence[TraceBlock],
    intervention_spec: Dict[str, Any],
    intervention_variant: str,
    pairs_to_generate_k: int,
    exec_token_limit: int,
    style_window_pairs: int,
    validation_feedback: str = "",
) -> Dict[str, Any]:
    cut_after_pairs = pair_count(prefix_blocks)
    next_original_steer, next_original_exec_preview = get_next_original_context(
        all_blocks, cut_after_pairs
    )

    return {
        "task_id": record.get("task_id", ""),
        "intervention_name": intervention_spec["name"],
        "intervention_variant": intervention_variant,
        "pairs_to_generate_k": pairs_to_generate_k,
        "exec_token_limit": exec_token_limit,
        "trace_prefix": render_trace(prefix_blocks, wrap_think=False),
        "local_style_window": build_style_window(
            prefix_blocks, style_window_pairs=style_window_pairs
        ),
        "next_original_steer": next_original_steer
        or "[no next original steer available]",
        "next_original_exec_preview": next_original_exec_preview or "[none]",
        "notes_on_tone": record.get(
            "notes_on_tone",
            "Steers should be short and directive. Exec blocks should be concrete, problem-directed, and avoid generic meta commentary.",
        ),
        "optional_validation_feedback": validation_feedback.strip(),
    }


def mock_intervention_window(intervention_variant: str, k: int) -> Dict[str, Any]:
    blocks = []
    for i in range(k):
        steer_text = intervention_variant if i == 0 else f"continue the local check {i}"
        exec_text = (
            "This is a short mock exec block used for offline testing of the augmentation "
            "pipeline. It preserves interleaving and stays well below the token cap."
        )
        blocks.append({"type": "steer", "text": steer_text})
        blocks.append({"type": "exec", "text": exec_text})
    return {"blocks": blocks}


def choose_prompt_path(
    prompts_dir: str | Path,
    mode: str,
    intervention_spec: Mapping[str, Any] | None = None,
) -> Path:
    """Return the prompt template path for an intervention mode.

    Args:
        prompts_dir: Directory containing prompt templates.
        mode: Requested editor mode.
        intervention_spec: Optional intervention spec that may override the prompt.

    Returns:
        Prompt template path.
    """

    return resolve_prompt_path(
        prompts_dir=prompts_dir,
        mode=mode,
        intervention_spec=intervention_spec,
    )


async def maybe_call_bridge_judge(
    *,
    openrouter: Optional[OpenRouterAsyncClient],
    prompts_dir: str | Path,
    trace_prefix: str,
    inserted_window: str,
    next_original_steer: str,
    next_original_exec_preview: str,
    mock: bool = False,
) -> BridgeJudgeResult:
    if mock or openrouter is None:
        # Conservative mock: ask whether the inserted window ends with obvious redirect language.
        lowered = inserted_window.lower()
        if any(
            s in lowered
            for s in [
                "backtrack",
                "different next move",
                "toy problem",
                "work backwards",
            ]
        ):
            return BridgeJudgeResult(
                decision="regenerate_suffix",
                reason="mock heuristic detected redirect-like language",
            )
        return BridgeJudgeResult(
            decision="keep_suffix",
            reason="mock heuristic found no obvious incompatibility",
        )

    system_prompt = Path(prompts_dir, "judge_system.md").read_text(encoding="utf-8")
    user_prompt = render_prompt(
        Path(prompts_dir, "bridge_judge.md"),
        {
            "trace_prefix": trace_prefix,
            "inserted_window": inserted_window,
            "next_original_steer": next_original_steer or "[none]",
            "next_original_exec_preview": next_original_exec_preview or "[none]",
        },
    )
    raw = await openrouter.chat_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_schema=build_bridge_judge_schema(),
    )
    return BridgeJudgeResult.model_validate(raw)


def splice_blocks(
    all_blocks: Sequence[TraceBlock],
    prefix_blocks: Sequence[TraceBlock],
    new_window_blocks: Sequence[TraceBlock],
    keep_suffix: bool,
) -> Tuple[List[TraceBlock], List[TraceBlock]]:
    prefix_plus_window = list(prefix_blocks) + list(new_window_blocks)
    if keep_suffix:
        suffix = list(all_blocks[len(prefix_blocks) :])
        return prefix_plus_window, prefix_plus_window + suffix
    return prefix_plus_window, []


def make_regen_seed(
    record: Dict[str, Any],
    augmented_prefix_trace: str,
    *,
    wrap_think: bool,
    vllm_base_url: Optional[str],
    vllm_model: Optional[str],
) -> Dict[str, Any]:
    prefix_blocks = parse_trace_text(augmented_prefix_trace)
    assistant_prefill = (
        render_open_exec_trace(prefix_blocks)
        if wrap_think
        else render_trace(prefix_blocks, wrap_think=False)
    )
    return {
        "user_prompt": record.get("user_prompt", ""),
        "assistant_prefill": assistant_prefill,
        "vllm_base_url": vllm_base_url or "http://localhost:8000/v1",
        "vllm_model": vllm_model or "",
        "note": (
            "This is a continuation seed only. Continue from inside the open think prefix. "
            "Use the original model and serving stack that match your training setup."
        ),
    }


async def augment_record(
    *,
    record: Dict[str, Any],
    intervention_spec: Dict[str, Any],
    prompts_dir: str | Path,
    rng: random.Random,
    token_counter: TokenCounter,
    exec_token_limit: int,
    style_window_pairs: int,
    openrouter: Optional[OpenRouterAsyncClient],
    mock_intervention: bool,
    wrap_think: bool,
    vllm_base_url: Optional[str],
    vllm_model: Optional[str],
    max_attempts: int = 3,
    run_bridge_judge: bool = True,
) -> Dict[str, Any]:
    trace_text = extract_trace_text(record)
    all_blocks = parse_trace_text(trace_text)

    input_errors = ensure_interleaved(
        all_blocks,
        token_counter=token_counter,
        exec_token_limit=exec_token_limit,
    )
    if input_errors:
        raise ValueError("Input trace validation failed: " + "; ".join(input_errors))

    total_pairs = pair_count(all_blocks)
    cut_after_pairs = select_cut_pair_index(
        total_pairs=total_pairs,
        preferred_slots=intervention_spec.get("preferred_slots", []),
        rng=rng,
    )
    prefix_blocks = slice_pairs(all_blocks, cut_after_pairs)

    chosen_mode = choose_mode(intervention_spec, rng)
    chosen_variant = choose_variant(intervention_spec, rng)
    k_pairs = choose_pairs_to_generate(intervention_spec, rng)

    system_prompt = Path(prompts_dir, "system.md").read_text(encoding="utf-8")
    prompt_path = choose_prompt_path(prompts_dir, chosen_mode)

    validation_feedback = ""
    generated_blocks: List[TraceBlock] = []
    generation_errors: List[str] = []
    raw_window_obj: Dict[str, Any] = {}

    for attempt in range(1, max_attempts + 1):
        values = build_prompt_values(
            record=record,
            all_blocks=all_blocks,
            prefix_blocks=prefix_blocks,
            intervention_spec=intervention_spec,
            intervention_variant=chosen_variant,
            pairs_to_generate_k=k_pairs,
            exec_token_limit=exec_token_limit,
            style_window_pairs=style_window_pairs,
            validation_feedback=validation_feedback,
        )
        user_prompt = render_prompt(prompt_path, values)

        if mock_intervention:
            raw_window_obj = mock_intervention_window(chosen_variant, k_pairs)
        else:
            if openrouter is None:
                raise RuntimeError(
                    "OpenRouter client is required unless mock_intervention is enabled."
                )
            raw_window_obj = await openrouter.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=build_intervention_schema(k_pairs),
            )

        generated_blocks, generation_errors = validate_generated_window(
            raw_window_obj,
            requested_pairs=k_pairs,
            required_first_steer=chosen_variant,
            token_counter=token_counter,
            exec_token_limit=exec_token_limit,
        )
        if not generation_errors:
            break

        validation_feedback = (
            "\n## Validation feedback from the previous attempt\n"
            "Revise the intervention window so that it satisfies all constraints.\n"
            + "\n".join(f"- {err}" for err in generation_errors)
        )

    if generation_errors:
        raise ValueError(
            "Generated intervention failed validation: " + "; ".join(generation_errors)
        )

    next_original_steer, next_original_exec_preview = get_next_original_context(
        all_blocks, cut_after_pairs
    )
    inserted_window_text = render_window(generated_blocks)

    keep_suffix = False
    suffix_decision = "regenerate_suffix"
    bridge_judge_reason = ""

    post_policy = intervention_spec["post_splice_policy"]
    if chosen_mode == "insert":
        keep_suffix = True
        suffix_decision = "keep_suffix"
        bridge_judge_reason = "insert mode chosen; keep suffix by construction"
    elif post_policy == "keep_original_suffix":
        keep_suffix = True
        suffix_decision = "keep_suffix"
        bridge_judge_reason = "post_splice_policy=keep_original_suffix"
    elif (
        chosen_mode == "bridge"
        and post_policy == "keep_if_next_steer_still_fits_else_regenerate"
    ):
        if run_bridge_judge:
            judge = await maybe_call_bridge_judge(
                openrouter=openrouter,
                prompts_dir=prompts_dir,
                trace_prefix=render_trace(prefix_blocks, wrap_think=False),
                inserted_window=inserted_window_text,
                next_original_steer=next_original_steer,
                next_original_exec_preview=next_original_exec_preview,
                mock=mock_intervention,
            )
            keep_suffix = judge.decision == "keep_suffix"
            suffix_decision = judge.decision
            bridge_judge_reason = judge.reason
        else:
            keep_suffix = False
            suffix_decision = "regenerate_suffix"
            bridge_judge_reason = "bridge judge disabled"
    else:
        keep_suffix = False
        suffix_decision = "regenerate_suffix"
        bridge_judge_reason = "redirective or regen-required policy"

    augmented_prefix_blocks, augmented_full_blocks = splice_blocks(
        all_blocks=all_blocks,
        prefix_blocks=prefix_blocks,
        new_window_blocks=generated_blocks,
        keep_suffix=keep_suffix,
    )

    augmented_prefix_trace = render_trace(
        augmented_prefix_blocks, wrap_think=wrap_think
    )
    augmented_full_trace = (
        render_trace(augmented_full_blocks, wrap_think=wrap_think)
        if augmented_full_blocks
        else None
    )
    prefix_pairs_to_show = min(2, pair_count(prefix_blocks))
    prefix_start_pair = max(0, pair_count(prefix_blocks) - prefix_pairs_to_show)
    prefix_pair_preview = [
        trace_block_to_dict(b)
        for b in slice_pair_blocks(
            prefix_blocks, start_pair=prefix_start_pair, num_pairs=prefix_pairs_to_show
        )
    ]
    next_suffix_pair_preview = []
    if keep_suffix and cut_after_pairs < total_pairs:
        next_suffix_pair_preview = [
            trace_block_to_dict(b)
            for b in slice_pair_blocks(
                all_blocks, start_pair=cut_after_pairs, num_pairs=1
            )
        ]

    result = dict(record)
    result["augmentation"] = {
        "intervention_name": intervention_spec["name"],
        "intervention_variant": chosen_variant,
        "category": intervention_spec.get("category", ""),
        "editor_mode": chosen_mode,
        "pairs_generated": k_pairs,
        "cut_after_pairs": cut_after_pairs,
        "total_pairs_before": total_pairs,
        "preferred_slots": intervention_spec.get("preferred_slots", []),
        "post_splice_policy": post_policy,
        "suffix_decision": suffix_decision,
        "bridge_judge_reason": bridge_judge_reason,
        "token_counter_method": token_counter.method,
        "exec_token_limit": exec_token_limit,
        "mode_weight_hint": intervention_spec.get("mode_weight_hint", {}),
        "why": intervention_spec.get("why", ""),
        "pair_preview": {
            "prefix_pairs": prefix_pair_preview,
            "suffix_pair_if_kept": next_suffix_pair_preview,
            "prefix_pairs_shown": min(2, pair_count(prefix_blocks)),
            "suffix_shown": keep_suffix and bool(next_suffix_pair_preview),
        },
    }
    result["generated_intervention_blocks"] = [
        {"type": b.type, "text": b.text} for b in generated_blocks
    ]
    result["augmented_prefix_trace"] = augmented_prefix_trace
    result["augmented_full_trace"] = augmented_full_trace
    result["regen_seed"] = make_regen_seed(
        record=record,
        augmented_prefix_trace=render_trace(augmented_prefix_blocks, wrap_think=False),
        wrap_think=wrap_think,
        vllm_base_url=vllm_base_url,
        vllm_model=vllm_model,
    )
    return result


def choose_intervention(
    interventions_obj: Dict[str, Any], rng: random.Random, name: Optional[str] = None
) -> Dict[str, Any]:
    interventions = interventions_obj["interventions"]
    if name:
        for item in interventions:
            if item["name"] == name:
                return item
        raise KeyError(f"Unknown intervention name: {name}")
    return rng.choice(interventions)
