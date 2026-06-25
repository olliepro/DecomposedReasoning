"""Run user-only story continuation prompts against a served vLLM model.

Example:
    python Analysis/scripts/run_story_completion_batch.py \
        --model-path /path/to/checkpoint \
        --base-url http://127.0.0.1:8042/v1
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import requests
from transformers import AutoTokenizer

DEFAULT_STORY_PROMPTS: tuple[str, ...] = (
    (
        "Complete this simple story in a coherent way.\n\n"
        "Nina found a locked suitcase under her late grandfather's bed. "
        "Inside was a train ticket dated tomorrow."
    ),
    (
        "Complete this simple story in a coherent way.\n\n"
        "The power went out during dinner, and when the lights came back, "
        "there was an extra place setting on the table."
    ),
    (
        "Complete this simple story in a coherent way.\n\n"
        "Every morning, the baker wrote one sentence on the shop window. "
        "Today it said, 'Do not trust the man with the red umbrella.'"
    ),
    (
        "Complete this simple story in a coherent way.\n\n"
        "Jonah's dog refused to enter the woods behind the school until the "
        "day Jonah heard someone whisper his own name from inside."
    ),
    (
        "Complete this simple story in a coherent way.\n\n"
        "At the town yard sale, Leila bought a camera with only three "
        "photographs left on the roll, and all three were of her front porch."
    ),
)


@dataclass(frozen=True)
class StoryPrompt:
    """One user-only story prompt ready for chat templating.

    Inputs:
        prompt_id: Stable prompt index in the output set.
        user_prompt: Raw user-only text prompt.

    Outputs:
        Serializable prompt record stored in the JSONL artifact.
    """

    prompt_id: int
    user_prompt: str


@dataclass(frozen=True)
class StoryCompletion:
    """One generated story continuation and its request metadata.

    Inputs:
        prompt: Source story prompt.
        rendered_prompt: Chat-templated prompt sent to the model.
        prompt_token_count: Token count of the rendered prompt.
        generated_text: Newly generated continuation.
        finish_reason: vLLM finish reason.
        stop_reason: Optional stop reason from the API.

    Outputs:
        JSON-serializable result row and HTML viewer content.

    Example:
        row = StoryCompletion(
            prompt=StoryPrompt(prompt_id=0, user_prompt="Finish this story..."),
            rendered_prompt="<|im_start|>user\n...\n<|im_start|>assistant",
            prompt_token_count=42,
            generated_text="The train ticket had her name on it.",
            finish_reason="length",
            stop_reason=None,
        )
    """

    prompt: StoryPrompt
    rendered_prompt: str
    prompt_token_count: int
    generated_text: str
    finish_reason: str | None
    stop_reason: str | None

    def to_json_dict(self) -> dict[str, Any]:
        """Return the canonical serialized representation."""
        data = asdict(self.prompt)
        data.update(
            rendered_prompt=self.rendered_prompt,
            prompt_token_count=self.prompt_token_count,
            generated_text=self.generated_text,
            generated_char_count=len(self.generated_text),
            finish_reason=self.finish_reason,
            stop_reason=self.stop_reason,
            completion_preview=preview_text(text=self.generated_text),
        )
        return data


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the story batch runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--async-concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--results-path", default="/tmp/story_completions.jsonl")
    parser.add_argument("--markdown-path", default="/tmp/story_completions_view.md")
    parser.add_argument("--html-path", default="/tmp/story_completions_view.html")
    return parser.parse_args()


def build_story_prompts() -> list[StoryPrompt]:
    """Build the default story prompt set.

    Outputs:
        Stable list of story prompts used for this run.

    Example:
        prompts = build_story_prompts()
    """
    return [
        StoryPrompt(prompt_id=prompt_id, user_prompt=user_prompt)
        for prompt_id, user_prompt in enumerate(DEFAULT_STORY_PROMPTS)
    ]


def render_prompt(*, tokenizer: Any, user_prompt: str) -> tuple[str, int]:
    """Render a user-only prompt with the model chat template.

    Inputs:
        tokenizer: Hugging Face tokenizer with a chat template.
        user_prompt: Raw user message only.

    Outputs:
        Rendered prompt text and prompt token count.
    """
    rendered_prompt = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    ).rstrip()
    prompt_token_count = len(
        tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
    )
    return rendered_prompt, prompt_token_count


def request_completion(
    *,
    base_url: str,
    model_path: str,
    rendered_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[str, str | None, str | None]:
    """Run one completion request against vLLM.

    Inputs:
        rendered_prompt: Chat-templated prompt text.
        seed: Deterministic request seed.

    Outputs:
        Generated text, finish reason, and optional stop reason.
    """
    payload = {
        "model": model_path,
        "prompt": rendered_prompt,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
        "seed": seed,
    }
    response = requests.post(url=f"{base_url}/completions", json=payload, timeout=3600)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    return str(choice["text"]), choice.get("finish_reason"), choice.get("stop_reason")


async def complete_prompts_async(
    *,
    prompts: Sequence[StoryPrompt],
    tokenizer: Any,
    model_path: str,
    base_url: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    async_concurrency: int,
    seed: int,
) -> list[StoryCompletion]:
    """Generate story continuations concurrently with async request batching.

    Inputs:
        prompts: User-only prompts to render and complete.
        async_concurrency: Max simultaneous requests.

    Outputs:
        Ordered list of story completions matching the prompt order.

    Example:
        rows = asyncio.run(
            complete_prompts_async(
                prompts=build_story_prompts(),
                tokenizer=tokenizer,
                model_path="/checkpoint",
                base_url="http://127.0.0.1:8042/v1",
                max_new_tokens=384,
                temperature=0.6,
                top_p=0.95,
                async_concurrency=4,
                seed=20260421,
            )
        )
    """
    semaphore = asyncio.Semaphore(async_concurrency)
    completion_map: dict[int, StoryCompletion] = {}

    async def run_one(prompt: StoryPrompt) -> None:
        rendered_prompt, prompt_token_count = render_prompt(
            tokenizer=tokenizer,
            user_prompt=prompt.user_prompt,
        )
        async with semaphore:
            generated_text, finish_reason, stop_reason = await asyncio.to_thread(
                request_completion,
                base_url=base_url,
                model_path=model_path,
                rendered_prompt=rendered_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed + prompt.prompt_id,
            )
        completion_map[prompt.prompt_id] = StoryCompletion(
            prompt=prompt,
            rendered_prompt=rendered_prompt,
            prompt_token_count=prompt_token_count,
            generated_text=generated_text,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )

    await asyncio.gather(*(run_one(prompt=prompt) for prompt in prompts))
    return [completion_map[prompt.prompt_id] for prompt in prompts]


def preview_text(*, text: str, limit: int = 160) -> str:
    """Return a compact preview snippet for markdown tables."""
    return " ".join(text.split())[:limit]


def write_results(*, path: Path, rows: Sequence[StoryCompletion]) -> None:
    """Write story completion rows to JSONL."""
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json_dict()))
            handle.write("\n")


def render_markdown(
    *, path: Path, rows: Sequence[StoryCompletion], model_path: str
) -> None:
    """Render a concise markdown summary for the story completions."""
    lines = [
        "# Story completions",
        "",
        f"Model: `{model_path}`  ",
        "",
        "| id | prompt toks | finish | prompt preview | completion preview |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        prompt_preview = preview_text(text=row.prompt.user_prompt)
        completion_preview = preview_text(text=row.generated_text)
        lines.append(
            "| {prompt_id} | {prompt_toks} | {finish_reason} | {prompt_preview} | {completion_preview} |".format(
                prompt_id=row.prompt.prompt_id,
                prompt_toks=row.prompt_token_count,
                finish_reason=row.finish_reason,
                prompt_preview=prompt_preview.replace("|", "\\|"),
                completion_preview=completion_preview.replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines))


def render_html(
    *, path: Path, rows: Sequence[StoryCompletion], model_path: str
) -> None:
    """Render a simple HTML viewer for the story completions."""
    cards: list[str] = []
    for row in rows:
        cards.append(f"""
            <section class='card'>
              <div class='meta'>
                <span>prompt {row.prompt.prompt_id}</span>
                <span>{row.prompt_token_count} prompt toks</span>
                <span>{html.escape(str(row.finish_reason))}</span>
              </div>
              <h2>User prompt</h2>
              <pre>{html.escape(row.prompt.user_prompt)}</pre>
              <h2>Completion</h2>
              <pre>{html.escape(row.generated_text)}</pre>
            </section>
            """)
    document = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset='utf-8'>
      <title>Story completions</title>
      <style>
        :root {{ color-scheme: light; }}
        body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 0; background: #f5f7fb; color: #16202a; }}
        main {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
        h1 {{ margin: 0 0 8px; font-size: 28px; }}
        p {{ margin: 0 0 20px; color: #44515f; }}
        .grid {{ display: grid; gap: 18px; }}
        .card {{ background: white; border: 1px solid #d6deea; border-radius: 14px; padding: 18px; box-shadow: 0 4px 18px rgba(20, 31, 48, 0.06); }}
        .meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
        .meta span {{ background: #eef3ff; color: #2e466e; border-radius: 999px; padding: 4px 10px; font-size: 12px; }}
        h2 {{ margin: 12px 0 8px; font-size: 16px; }}
        pre {{ white-space: pre-wrap; word-break: break-word; background: #f8fafc; border-radius: 10px; padding: 12px; border: 1px solid #e3e8f0; line-height: 1.45; font-size: 13px; margin: 0; }}
      </style>
    </head>
    <body>
      <main>
        <h1>Story completions</h1>
        <p>Model: {html.escape(model_path)}. User-only prompts with no injected assistant prefix.</p>
        <div class='grid'>
          {''.join(cards)}
        </div>
      </main>
    </body>
    </html>
    """
    path.write_text(document)


def main() -> None:
    """Run the story completion sweep and write `/tmp` artifacts."""
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    prompts = build_story_prompts()
    rows = asyncio.run(
        complete_prompts_async(
            prompts=prompts,
            tokenizer=tokenizer,
            model_path=args.model_path,
            base_url=args.base_url,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            async_concurrency=args.async_concurrency,
            seed=args.seed,
        )
    )
    write_results(path=Path(args.results_path), rows=rows)
    render_markdown(
        path=Path(args.markdown_path), rows=rows, model_path=args.model_path
    )
    render_html(path=Path(args.html_path), rows=rows, model_path=args.model_path)


if __name__ == "__main__":
    main()
