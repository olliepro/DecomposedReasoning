from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Iterator, cast

from google import genai
from google.genai import types
from tqdm import tqdm

from pipeline_common import count_jsonl_rows, extract_think_blocks, iter_jsonl, write_jsonl
from pipeline_types import PromptConfig, ThinkTask, TransformConfig


def clean_model_output(text: str) -> str:
    """Normalize model output by stripping fences and surrounding text.

    Args:
        text: Raw model output.

    Returns:
        Cleaned output.

    Example:
        >>> clean_model_output("```xml\\n<a/>\\n```")
        '<a/>'
    """
    cleaned = text.lstrip("\ufeff").strip()
    if "```" in cleaned:
        cleaned = cleaned[cleaned.find("```") :]
    cleaned = re.sub(r"^```(?:[a-zA-Z0-9_-]+)?\s*", "", cleaned)
    if "```" in cleaned:
        cleaned = cleaned[: cleaned.rfind("```")]
    return cleaned.strip()


def extract_response_text(response: object) -> str:
    """Extract textual content from a GenAI response object.

    Args:
        response: GenAI response object.

    Returns:
        Extracted text.
    """
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""
    first = candidates[0]
    content = getattr(first, "content", None)
    parts = getattr(content, "parts", None) or []
    texts = [
        getattr(part, "text", "")
        for part in parts
        if isinstance(getattr(part, "text", None), str)
    ]
    return "".join(texts)


def normalize_thinking_level(level: str | None) -> str | None:
    """Normalize thinking-level value.

    Args:
        level: Raw level string.

    Returns:
        Uppercased level or None when disabled.
    """
    if level is None:
        return None
    if level.lower() == "none":
        return None
    return level.strip().upper()


def validate_thinking_level(model_id: str, thinking_level: str | None) -> str | None:
    """Validate model/thinking-level compatibility.

    Args:
        model_id: Model identifier.
        thinking_level: Normalized thinking level.

    Returns:
        Validated thinking level.
    """
    if thinking_level is None:
        return None
    if "gemini-3" not in model_id:
        raise SystemExit("--thinking-level is only supported for Gemini 3 models.")
    if thinking_level in {"MINIMAL", "MEDIUM"} and "flash" not in model_id:
        raise SystemExit("MINIMAL/MEDIUM thinking levels require a Gemini 3 Flash model.")
    return thinking_level


def resolve_api_key(raw_key: str | None, env_values: dict[str, str], dry_run: bool) -> str:
    """Resolve API key for transform stage.

    Args:
        raw_key: Optional CLI key.
        env_values: Parsed dotenv values.
        dry_run: If true, allow placeholder key.

    Returns:
        API key.
    """
    if raw_key:
        return raw_key
    candidates = [
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("VERTEX_KEY"),
        os.getenv("VERTEX_API_KEY"),
        os.getenv("GOOGLE_CLOUD_API_KEY"),
        env_values.get("GEMINI_API_KEY"),
        env_values.get("GOOGLE_API_KEY"),
        env_values.get("VERTEX_KEY"),
        env_values.get("VERTEX_API_KEY"),
        env_values.get("GOOGLE_CLOUD_API_KEY"),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    if dry_run:
        return "DRY_RUN_KEY"
    raise SystemExit("API key is required. Use --api-key or set VERTEX_KEY/GEMINI_API_KEY.")


def load_prompt_text(path: Path) -> str:
    """Load a prompt file.

    Args:
        path: Prompt path.

    Returns:
        Prompt text.
    """
    if not path.exists():
        raise SystemExit(f"Missing prompt file: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"Prompt file is empty: {path}")
    return text


def render_user_prompt(template: str, think_text: str) -> str:
    """Render user template with think text.

    Args:
        template: Prompt template containing `{think_text}`.
        think_text: Think block content.

    Returns:
        Rendered prompt.
    """
    if "{think_text}" not in template:
        raise SystemExit("user_prompt.md must include `{think_text}` placeholder.")
    return template.replace("{think_text}", think_text)


def build_client(config: TransformConfig) -> genai.Client:
    """Create GenAI client.

    Args:
        config: Transform config.

    Returns:
        Configured client.
    """
    if config.mode == "gemini":
        return genai.Client(api_key=config.api_key)

    http_options = types.HttpOptions(apiVersion="v1")
    if config.mode == "vertex":
        if not config.project_id or not config.location:
            raise SystemExit("--project and --location are required in vertex mode.")
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        os.environ["GOOGLE_CLOUD_PROJECT"] = config.project_id
        os.environ["GOOGLE_CLOUD_LOCATION"] = config.location

    return genai.Client(vertexai=True, api_key=config.api_key, http_options=http_options)


def call_model(
    client: genai.Client,
    config: TransformConfig,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call non-batch model API.

    Args:
        client: GenAI client.
        config: Transform config.
        system_prompt: System prompt text.
        user_prompt: User prompt text.

    Returns:
        Cleaned model output.
    """
    thinking_config = (
        types.ThinkingConfig(thinking_level=config.thinking_level)
        if config.thinking_level
        else None
    )
    gen_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        thinking_config=thinking_config,
    )
    response = client.models.generate_content(
        model=config.model_id,
        contents=user_prompt,
        config=gen_config,
    )
    return clean_model_output(response.text or "")


def transform_think_block(
    client: genai.Client,
    config: TransformConfig,
    system_prompt: str,
    user_template: str,
    think_text: str,
) -> str:
    """Transform one think block with retries.

    Args:
        client: GenAI client.
        config: Transform config.
        system_prompt: System prompt text.
        user_template: User prompt template.
        think_text: Raw think block.

    Returns:
        Transformed content.
    """
    user_prompt = render_user_prompt(template=user_template, think_text=think_text)
    last_error: Exception | None = None
    for attempt in range(1, config.retry_limit + 1):
        try:
            return call_model(
                client=client,
                config=config,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(config.retry_sleep_seconds * attempt)
    raise RuntimeError("Model call failed after retries") from last_error


def replace_think_blocks(content: str, replacements: Iterator[str]) -> str:
    """Replace think blocks in one message.

    Args:
        content: Assistant message content.
        replacements: Iterator of replacement strings.

    Returns:
        Updated content with replaced think blocks.
    """

    def _replace(match: re.Match[str]) -> str:
        block = match.group(1)
        if not block.strip():
            return "<think>\n\n</think>"
        try:
            transformed = clean_model_output(next(replacements))
        except StopIteration as exc:
            raise RuntimeError("Ran out of transformed outputs.") from exc
        return f"<think>\n{transformed}\n</think>"

    return re.sub(r"<think>(.*?)</think>", _replace, content, flags=re.DOTALL | re.IGNORECASE)


def resolve_batch_model_id(model_id: str) -> str:
    """Normalize model ID for batch API.

    Args:
        model_id: Model identifier.

    Returns:
        Batch-ready model identifier.
    """
    if model_id.startswith("models/") or model_id.startswith("projects/"):
        return model_id
    return f"models/{model_id}"


def collect_transform_rows(
    source_path: Path,
    seen_ids: set[str],
    max_needed: int,
) -> list[dict[str, object]]:
    """Collect rows for transform stage, skipping seen IDs.

    Args:
        source_path: Stratified input path.
        seen_ids: Already transformed row IDs.
        max_needed: Max rows needed.

    Returns:
        Rows selected for transform.
    """
    rows: list[dict[str, object]] = []
    for row in iter_jsonl(path=source_path):
        row_id = row.get("id")
        if isinstance(row_id, str) and row_id in seen_ids:
            continue
        rows.append(row)
        if len(rows) >= max_needed:
            break
    return rows


def load_seen_ids(path: Path) -> set[str]:
    """Load transformed row IDs from output file.

    Args:
        path: Transformed output path.

    Returns:
        Set of seen row IDs.
    """
    if not path.exists():
        return set()
    seen: set[str] = set()
    for row in iter_jsonl(path=path):
        row_id = row.get("id")
        if isinstance(row_id, str):
            seen.add(row_id)
    return seen


def count_think_blocks(rows: list[dict[str, object]]) -> int:
    """Count total think blocks across rows.

    Args:
        rows: Row buffer.

    Returns:
        Think block count.
    """
    total = 0
    for row in rows:
        messages = row.get("messages", [])
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str):
                total += len(extract_think_blocks(text=content))
    return total


def build_inline_batch_requests(
    rows: list[dict[str, object]],
    system_prompt: str,
    user_template: str,
    config: TransformConfig,
) -> tuple[list[dict[str, object]], list[ThinkTask]]:
    """Build inline batch requests and task map.

    Args:
        rows: Rows to transform.
        system_prompt: System prompt text.
        user_template: User prompt template.
        config: Transform config.

    Returns:
        Requests and think-task mapping.
    """
    requests: list[dict[str, object]] = []
    tasks: list[ThinkTask] = []
    for row_index, row in enumerate(rows):
        messages = row.get("messages", [])
        if not isinstance(messages, list):
            continue
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            blocks = extract_think_blocks(text=content)
            for block_index, block in enumerate(blocks):
                request_index = len(tasks)
                user_prompt = render_user_prompt(template=user_template, think_text=block)
                request_config: dict[str, object] = {
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "temperature": config.temperature,
                    "max_output_tokens": config.max_output_tokens,
                }
                if config.thinking_level:
                    request_config["thinking_config"] = {"thinking_level": config.thinking_level}
                requests.append(
                    {
                        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                        "metadata": {"request_index": str(request_index)},
                        "config": request_config,
                    }
                )
                tasks.append(
                    ThinkTask(
                        row_index=row_index,
                        message_index=message_index,
                        block_index=block_index,
                    )
                )
    return requests, tasks


def run_batch_requests(
    client: genai.Client,
    config: TransformConfig,
    inline_requests: list[dict[str, object]],
) -> list[str]:
    """Run batch API and return outputs.

    Args:
        client: GenAI client.
        config: Transform config.
        inline_requests: Inline request payloads.

    Returns:
        Cleaned outputs in request order.
    """
    if not inline_requests:
        return []

    batch_job = client.batches.create(
        model=resolve_batch_model_id(model_id=config.model_id),
        src=inline_requests,
        config={"display_name": f"build-sft-{int(time.time())}"},
    )
    assert batch_job.name is not None
    print(f"Batch submitted: name={batch_job.name}", flush=True)

    poll_count = 0
    while True:
        batch_job = client.batches.get(name=batch_job.name)
        state_obj = getattr(batch_job, "state", None)
        state_name = getattr(state_obj, "name", str(state_obj))
        poll_count += 1
        if poll_count == 1 or poll_count % 12 == 0:
            print(f"Batch status: state={state_name} polls={poll_count}", flush=True)
        if state_name in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}:
            break
        time.sleep(config.batch_poll_seconds)

    if state_name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch job ended with {state_name}")

    destination = getattr(batch_job, "dest", None)
    responses = getattr(destination, "inlined_responses", None)
    if not responses:
        raise RuntimeError("Batch job succeeded without inline responses.")

    outputs: list[str] = [""] * len(inline_requests)
    seen_indexes: set[int] = set()
    failed_indexes: set[int] = set()
    failed_reasons: dict[int, str] = {}

    for item in responses:
        metadata = getattr(item, "metadata", None)
        request_index_raw = metadata.get("request_index") if isinstance(metadata, dict) else None
        if request_index_raw is None:
            raise RuntimeError("Batch response item is missing request_index metadata.")
        try:
            request_index = int(request_index_raw)
        except ValueError as exc:
            raise RuntimeError(f"Invalid request_index metadata: {request_index_raw}") from exc
        if request_index < 0 or request_index >= len(outputs):
            raise RuntimeError(f"Out-of-range request_index metadata: {request_index}")
        if request_index in seen_indexes:
            raise RuntimeError(f"Duplicate request_index metadata: {request_index}")

        response_error = getattr(item, "error", None)
        if response_error is not None:
            failed_indexes.add(request_index)
            failed_reasons[request_index] = str(response_error)
            continue
        response = getattr(item, "response", None)
        outputs[request_index] = clean_model_output(
            extract_response_text(response=response)
        )
        seen_indexes.add(request_index)

    all_indexes = set(range(len(outputs)))
    missing_indexes = sorted(all_indexes - seen_indexes - failed_indexes)
    if missing_indexes:
        for missing_index in missing_indexes:
            failed_indexes.add(missing_index)
            failed_reasons[missing_index] = "Missing inlined response for request index."

    if failed_indexes:
        print(
            f"Retrying {len(failed_indexes)} failed batch item(s) with direct calls.",
            flush=True,
        )
        for failed_index in sorted(failed_indexes):
            inline_request = inline_requests[failed_index]
            inline_contents = cast(Any, inline_request.get("contents"))
            inline_config = cast(Any, inline_request.get("config"))
            last_error: Exception | None = None
            for attempt in range(1, config.retry_limit + 1):
                try:
                    direct_response = client.models.generate_content(
                        model=config.model_id,
                        contents=inline_contents,
                        config=inline_config,
                    )
                    outputs[failed_index] = clean_model_output(
                        extract_response_text(response=direct_response)
                    )
                    seen_indexes.add(failed_index)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    time.sleep(config.retry_sleep_seconds * attempt)
            if failed_index not in seen_indexes:
                reason = failed_reasons.get(failed_index, "Unknown batch response error.")
                raise RuntimeError(
                    "Failed to recover batch item after retries. "
                    f"request_index={failed_index} batch_error={reason}"
                ) from last_error

    if len(seen_indexes) != len(outputs):
        unresolved_indexes = sorted(all_indexes - seen_indexes)
        raise RuntimeError(
            f"Unresolved batch outputs after retry. Missing indexes: {unresolved_indexes[:10]}"
        )

    return outputs


def apply_transforms_to_rows(
    rows: list[dict[str, object]],
    outputs: list[str],
) -> list[dict[str, object]]:
    """Apply transformed outputs to rows.

    Args:
        rows: Input rows.
        outputs: Transformed think outputs.

    Returns:
        Rows with updated assistant messages.
    """
    output_iter = iter(outputs)
    updated_rows: list[dict[str, object]] = []

    for row in rows:
        messages = row.get("messages", [])
        updated_messages: list[dict[str, object]] = []
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                updated_messages.append(message)
                continue
            content = message.get("content")
            if not isinstance(content, str):
                updated_messages.append(message)
                continue
            updated_content = replace_think_blocks(content=content, replacements=output_iter)
            updated_message = dict(message)
            updated_message["content"] = updated_content
            updated_messages.append(updated_message)

        output_row = dict(row)
        output_row["messages"] = updated_messages
        updated_rows.append(output_row)

    try:
        next(output_iter)
        raise RuntimeError("Received extra outputs not mapped to think blocks.")
    except StopIteration:
        pass

    return updated_rows


def run_non_batch_transform(
    client: genai.Client,
    config: TransformConfig,
    system_prompt: str,
    user_template: str,
    rows_to_transform: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Transform rows with direct model calls.

    Args:
        client: GenAI client.
        config: Transform config.
        system_prompt: System prompt text.
        user_template: User prompt template.
        rows_to_transform: Input rows.

    Returns:
        Transformed rows.
    """
    updated_rows: list[dict[str, object]] = []
    progress = tqdm(rows_to_transform, desc="Transforming rows", unit="row")

    for row in progress:
        messages = row.get("messages", [])
        updated_messages: list[dict[str, object]] = []
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                updated_messages.append(message)
                continue
            content = message.get("content")
            if not isinstance(content, str):
                updated_messages.append(message)
                continue
            blocks = extract_think_blocks(text=content)
            transformed_blocks = [
                transform_think_block(
                    client=client,
                    config=config,
                    system_prompt=system_prompt,
                    user_template=user_template,
                    think_text=block,
                )
                for block in blocks
            ]
            updated_content = replace_think_blocks(
                content=content,
                replacements=iter(transformed_blocks),
            )
            updated_message = dict(message)
            updated_message["content"] = updated_content
            updated_messages.append(updated_message)

        output_row = dict(row)
        output_row["messages"] = updated_messages
        updated_rows.append(output_row)

    return updated_rows


def write_transformed_rows(
    rows: list[dict[str, object]],
    output_path: Path,
    config: TransformConfig,
    prompts: PromptConfig,
) -> None:
    """Write transformed rows with transform metadata.

    Args:
        rows: Transformed rows.
        output_path: Output JSONL path.
        config: Transform config.
        prompts: Prompt config.
    """
    for row in rows:
        output_row = dict(row)
        output_row["transform_meta"] = {
            "model": config.model_id,
            "mode": config.mode,
            "batch": config.batch,
            "max_output_tokens": config.max_output_tokens,
            "thinking_level": config.thinking_level,
            "system_prompt_path": str(prompts.system_prompt_path),
            "user_prompt_path": str(prompts.user_prompt_path),
        }
        write_jsonl(output_path=output_path, row=output_row)


def run_transform_stage(
    config: TransformConfig,
    prompts: PromptConfig,
    stratified_path: Path,
    output_path: Path,
    resume: bool,
    dry_run: bool,
) -> dict[str, object]:
    """Run transform stage.

    Args:
        config: Transform config.
        prompts: Prompt config.
        stratified_path: Stratified sample path.
        output_path: Transform output path.
        resume: Resume from existing output.
        dry_run: Skip API calls/writes.

    Returns:
        Stage metadata.
    """
    if not stratified_path.exists():
        raise SystemExit(f"Missing stratified sample: {stratified_path}")

    existing_rows = count_jsonl_rows(path=output_path) if resume else 0
    rows_left = max(config.max_rows - existing_rows, 0)
    if rows_left == 0:
        return {
            "target_rows": config.max_rows,
            "existing_rows": existing_rows,
            "rows_left": 0,
            "skipped": True,
        }

    seen_ids = load_seen_ids(path=output_path) if resume else set()
    rows_to_transform = collect_transform_rows(
        source_path=stratified_path,
        seen_ids=seen_ids,
        max_needed=rows_left,
    )
    think_blocks = count_think_blocks(rows=rows_to_transform)

    if dry_run:
        return {
            "target_rows": config.max_rows,
            "existing_rows": existing_rows,
            "rows_left": rows_left,
            "rows_selected": len(rows_to_transform),
            "think_blocks": think_blocks,
            "batch": config.batch,
        }

    if not resume and output_path.exists():
        output_path.unlink()

    system_prompt = load_prompt_text(path=prompts.system_prompt_path)
    user_template = load_prompt_text(path=prompts.user_prompt_path)
    client = build_client(config=config)

    if config.batch:
        requests, _ = build_inline_batch_requests(
            rows=rows_to_transform,
            system_prompt=system_prompt,
            user_template=user_template,
            config=config,
        )
        outputs = run_batch_requests(
            client=client,
            config=config,
            inline_requests=requests,
        )
        updated_rows = apply_transforms_to_rows(rows=rows_to_transform, outputs=outputs)
    else:
        updated_rows = run_non_batch_transform(
            client=client,
            config=config,
            system_prompt=system_prompt,
            user_template=user_template,
            rows_to_transform=rows_to_transform,
        )

    write_transformed_rows(
        rows=updated_rows,
        output_path=output_path,
        config=config,
        prompts=prompts,
    )
    return {
        "target_rows": config.max_rows,
        "existing_rows": existing_rows,
        "rows_selected": len(rows_to_transform),
        "rows_emitted": len(updated_rows),
        "think_blocks": think_blocks,
        "batch": config.batch,
    }
