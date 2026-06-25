"""Run paired interventions over user-only story completions.

Example:
    python Analysis/scripts/run_story_intervention_batch.py \
        --model-path /path/to/checkpoint \
        --base-url http://127.0.0.1:8042/v1 \
        --intervention-phrase "List 5 potential approaches"
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import requests
from transformers import AutoTokenizer


@dataclass(frozen=True)
class StorySourceRow:
    """One saved user-only story completion row.

    Inputs:
        prompt_id: Stable story prompt index.
        user_prompt: Original user-only story prompt.
        rendered_prompt: Chat-templated prompt sent to the model.
        generated_text: Full assistant completion from the story sweep.

    Outputs:
        Source row used to extract one cut point for interventions.
    """

    prompt_id: int
    user_prompt: str
    rendered_prompt: str
    generated_text: str


@dataclass(frozen=True)
class StoryCutPoint:
    """One story completion cut immediately after a sampled `<steer>` tag.

    Inputs:
        prompt_id: Stable story prompt index.
        user_prompt: Original user-only story prompt.
        rendered_prompt: Chat-templated user prompt with open assistant turn.
        steer_index: Zero-based selected steer boundary within the completion.
        steer_count_total: Total steer boundaries found in the completion.
        partial_assistant_prefix: Assistant prefix ending right after `<steer>`.
        intervention_phrase: Text appended inside the open steer tag.

    Outputs:
        Serializable cut-point record used for paired continuation requests.
    """

    prompt_id: int
    user_prompt: str
    rendered_prompt: str
    steer_index: int
    steer_count_total: int
    partial_assistant_prefix: str
    partial_prefix_chars: int
    intervention_phrase: str


@dataclass(frozen=True)
class PromptVariant:
    """One prompt variant prepared for a paired async request.

    Inputs:
        prompt_id: Parent story prompt index.
        variant: Either `baseline` or `intervention`.
        prompt: Full prompt text sent to vLLM.
        prompt_token_count: Token count of the full prompt.
        assistant_prefix: Assistant-side prefix embedded before generation.

    Outputs:
        Metadata used to reconstruct paired results.
    """

    prompt_id: int
    variant: str
    prompt: str
    prompt_token_count: int
    assistant_prefix: str
    prompt_tail: str


@dataclass(frozen=True)
class PromptBatch:
    """One paired prompt-list request sent to vLLM.

    Inputs:
        prompt_id: Parent story prompt index.
        seed: Deterministic request seed.
        prompts: Ordered baseline/intervention variants for one story.

    Outputs:
        Batch metadata used to reconstruct final paired rows.
    """

    prompt_id: int
    seed: int
    prompts: tuple[PromptVariant, ...]


@dataclass(frozen=True)
class VariantResult:
    """One generated continuation for a story intervention variant.

    Inputs:
        prompt_token_count: Token count of the full prompt.
        finish_reason: vLLM finish reason.
        stop_reason: Optional stop reason from the API.
        assistant_prefix: Assistant prefix included before generation.
        generated_text: Newly generated continuation text only.

    Outputs:
        Structured result payload for JSONL and HTML rendering.
    """

    prompt_token_count: int
    finish_reason: str | None
    stop_reason: str | None
    assistant_prefix: str
    generated_text: str
    generated_char_count: int
    prompt_tail: str
    continuation_preview: str


@dataclass(frozen=True)
class PairedResult:
    """Baseline and intervention continuations for one story cut point.

    Inputs:
        cut_point: Source cut-point metadata.
        baseline: Continuation from the original story prefix.
        intervention: Continuation from the appended intervention phrase.

    Outputs:
        JSON-serializable paired comparison row.
    """

    cut_point: StoryCutPoint
    baseline: VariantResult
    intervention: VariantResult

    def to_json_dict(self) -> dict[str, Any]:
        """Return the canonical serialized representation for one paired row."""
        data = asdict(self.cut_point)
        data["baseline"] = asdict(self.baseline)
        data["intervention"] = asdict(self.intervention)
        return data


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the story intervention runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--intervention-phrase", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--async-concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--story-results-path", default="/tmp/story_completions.jsonl")
    parser.add_argument("--points-path", default="/tmp/story_intervention_points.jsonl")
    parser.add_argument(
        "--results-path",
        default="/tmp/story_intervention_continuations.jsonl",
    )
    parser.add_argument("--markdown-path", default="/tmp/story_intervention_view.md")
    parser.add_argument("--html-path", default="/tmp/story_intervention_view.html")
    return parser.parse_args()


def load_story_rows(*, path: Path) -> list[StorySourceRow]:
    """Load saved story completions into typed source rows."""
    with path.open() as handle:
        raw_rows = [json.loads(line) for line in handle]
    return [
        StorySourceRow(
            prompt_id=int(row["prompt_id"]),
            user_prompt=str(row["user_prompt"]),
            rendered_prompt=str(row["rendered_prompt"]),
            generated_text=str(row["generated_text"]),
        )
        for row in raw_rows
    ]


def steer_open_positions(*, text: str) -> list[int]:
    """Return indexes immediately after each `<steer>` open tag."""
    marker = "<steer>"
    start = 0
    positions: list[int] = []
    while True:
        index = text.find(marker, start)
        if index < 0:
            return positions
        positions.append(index + len(marker))
        start = index + len(marker)


def random_steer_cut_point(
    *,
    row: StorySourceRow,
    intervention_phrase: str,
    chooser: random.Random,
) -> StoryCutPoint:
    """Extract one random `<steer>` cut point from a story completion.

    Inputs:
        row: Story completion source row.
        intervention_phrase: Text appended inside the open steer tag.
        chooser: Deterministic RNG used to sample the steer boundary.

    Outputs:
        Typed cut-point record ending right after the sampled `<steer>`.

    Example:
        chooser = random.Random(7)
        cut = random_steer_cut_point(
            row=row,
            intervention_phrase="List options",
            chooser=chooser,
        )
    """
    positions = steer_open_positions(text=row.generated_text)
    assert positions, f"no <steer> tag found for prompt {row.prompt_id}"
    steer_index = chooser.randrange(len(positions))
    cut_index = positions[steer_index]
    prefix = row.generated_text[:cut_index]
    return StoryCutPoint(
        prompt_id=row.prompt_id,
        user_prompt=row.user_prompt,
        rendered_prompt=row.rendered_prompt,
        steer_index=steer_index,
        steer_count_total=len(positions),
        partial_assistant_prefix=prefix,
        partial_prefix_chars=len(prefix),
        intervention_phrase=intervention_phrase,
    )


def build_cut_points(
    *, rows: Sequence[StorySourceRow], intervention_phrase: str, seed: int
) -> list[StoryCutPoint]:
    """Extract one random steer cut point from each story completion row."""
    chooser = random.Random(seed)
    return [
        random_steer_cut_point(
            row=row,
            intervention_phrase=intervention_phrase,
            chooser=chooser,
        )
        for row in rows
    ]


def preview_text(*, text: str, limit: int = 160) -> str:
    """Return a compact preview snippet for markdown tables."""
    return " ".join(text.split())[:limit]


def count_prompt_tokens(*, tokenizer: Any, prompt: str) -> int:
    """Return model-tokenized prompt length.

    Inputs:
        tokenizer: Hugging Face tokenizer for the served model.
        prompt: Full prompt text sent to the model.

    Outputs:
        Token count computed from the model tokenizer.
    """
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def build_prompt_variants(
    *, cut_points: Sequence[StoryCutPoint], tokenizer: Any, max_new_tokens: int
) -> list[PromptVariant]:
    """Build baseline and intervention prompts for every story cut point."""
    variants: list[PromptVariant] = []
    for cut_point in cut_points:
        prefixes = (
            ("baseline", cut_point.partial_assistant_prefix),
            (
                "intervention",
                cut_point.partial_assistant_prefix + cut_point.intervention_phrase,
            ),
        )
        for variant_name, assistant_prefix in prefixes:
            prompt = cut_point.rendered_prompt + assistant_prefix
            prompt_token_count = count_prompt_tokens(
                tokenizer=tokenizer,
                prompt=prompt,
            )
            assert prompt_token_count + max_new_tokens <= 32768, prompt_token_count
            variants.append(
                PromptVariant(
                    prompt_id=cut_point.prompt_id,
                    variant=variant_name,
                    prompt=prompt,
                    prompt_token_count=prompt_token_count,
                    assistant_prefix=assistant_prefix,
                    prompt_tail=prompt[-500:],
                )
            )
    return variants


def build_prompt_batches(
    *, prompts: Sequence[PromptVariant], seed: int
) -> list[PromptBatch]:
    """Group flat prompt variants into paired async prompt-list requests."""
    grouped: dict[int, list[PromptVariant]] = {}
    for prompt in prompts:
        grouped.setdefault(prompt.prompt_id, []).append(prompt)
    batches: list[PromptBatch] = []
    for prompt_id in sorted(grouped):
        prompt_pair = tuple(sorted(grouped[prompt_id], key=lambda item: item.variant))
        assert len(prompt_pair) == 2, "expected baseline/intervention pair"
        batches.append(
            PromptBatch(prompt_id=prompt_id, seed=seed + prompt_id, prompts=prompt_pair)
        )
    return batches


def request_batch_results(
    *,
    base_url: str,
    model_path: str,
    prompt_batch: PromptBatch,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[VariantResult]:
    """Generate one paired story batch in a single prompt-list API call."""
    payload = {
        "model": model_path,
        "prompt": [prompt.prompt for prompt in prompt_batch.prompts],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
        "seed": prompt_batch.seed,
    }
    response = requests.post(url=f"{base_url}/completions", json=payload, timeout=3600)
    response.raise_for_status()
    choices = sorted(response.json()["choices"], key=lambda choice: choice["index"])
    assert len(choices) == len(
        prompt_batch.prompts
    ), "batched completion count mismatch"
    results: list[VariantResult] = []
    for prompt, choice in zip(prompt_batch.prompts, choices):
        generated_text = str(choice["text"])
        results.append(
            VariantResult(
                prompt_token_count=prompt.prompt_token_count,
                finish_reason=choice.get("finish_reason"),
                stop_reason=choice.get("stop_reason"),
                assistant_prefix=prompt.assistant_prefix,
                generated_text=generated_text,
                generated_char_count=len(generated_text),
                prompt_tail=prompt.prompt_tail,
                continuation_preview=preview_text(text=generated_text),
            )
        )
    return results


async def generate_batches_async(
    *,
    base_url: str,
    model_path: str,
    prompt_batches: Sequence[PromptBatch],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    async_concurrency: int,
) -> dict[tuple[int, str], VariantResult]:
    """Generate story intervention batches concurrently.

    Inputs:
        prompt_batches: Paired prompt batches, one request per story prompt.
        async_concurrency: Max simultaneous vLLM requests.

    Outputs:
        Mapping keyed by `(prompt_id, variant)` for stable reconstruction.
    """
    semaphore = asyncio.Semaphore(async_concurrency)
    result_map: dict[tuple[int, str], VariantResult] = {}

    async def run_one(prompt_batch: PromptBatch) -> None:
        async with semaphore:
            batch_results = await asyncio.to_thread(
                request_batch_results,
                base_url=base_url,
                model_path=model_path,
                prompt_batch=prompt_batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        for prompt, result in zip(prompt_batch.prompts, batch_results):
            result_map[(prompt.prompt_id, prompt.variant)] = result

    await asyncio.gather(
        *(run_one(prompt_batch=prompt_batch) for prompt_batch in prompt_batches)
    )
    return result_map


def pair_results(
    *,
    cut_points: Sequence[StoryCutPoint],
    result_map: dict[tuple[int, str], VariantResult],
) -> list[PairedResult]:
    """Reconstruct paired result rows from the async result mapping."""
    return [
        PairedResult(
            cut_point=cut_point,
            baseline=result_map[(cut_point.prompt_id, "baseline")],
            intervention=result_map[(cut_point.prompt_id, "intervention")],
        )
        for cut_point in cut_points
    ]


def write_jsonl_rows(*, path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write JSONL rows to disk."""
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def render_markdown(
    *, path: Path, rows: Sequence[PairedResult], model_path: str, base_url: str
) -> None:
    """Render a concise markdown summary for the story intervention results."""
    lines = [
        "# Story intervention continuations",
        "",
        "Mode: `async paired prompt batches`  ",
        f"Model: `{model_path}`  ",
        f"Base URL: `{base_url}`  ",
        "",
        "| id | steer | base toks | base finish | intervention toks | intervention finish |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {prompt_id} | {steer_index}/{steer_total} | {base_toks} | {base_finish} | {intervention_toks} | {intervention_finish} |".format(
                prompt_id=row.cut_point.prompt_id,
                steer_index=row.cut_point.steer_index,
                steer_total=row.cut_point.steer_count_total,
                base_toks=row.baseline.prompt_token_count,
                base_finish=row.baseline.finish_reason,
                intervention_toks=row.intervention.prompt_token_count,
                intervention_finish=row.intervention.finish_reason,
            )
        )
        lines.append("")
        lines.append(f"Prompt: `{preview_text(text=row.cut_point.user_prompt)}`")
        lines.append("")
        lines.append(f"Baseline preview: `{row.baseline.continuation_preview}`")
        lines.append("")
        lines.append(f"Intervention preview: `{row.intervention.continuation_preview}`")
        lines.append("")
    path.write_text("\n".join(lines))


def render_html(*, path: Path, rows: Sequence[PairedResult], model_path: str) -> None:
    """Render a side-by-side HTML viewer for the story interventions."""
    cards: list[str] = []
    for row in rows:
        cards.append(f"""
            <section class='card'>
              <div class='meta'>
                <span>story {row.cut_point.prompt_id}</span>
                <span>steer {row.cut_point.steer_index}/{row.cut_point.steer_count_total}</span>
                <span>{row.baseline.prompt_token_count} base toks</span>
                <span>{row.intervention.prompt_token_count} intervention toks</span>
              </div>
              <h2>User prompt</h2>
              <pre>{html.escape(row.cut_point.user_prompt)}</pre>
              <h2>Cut prefix tail</h2>
              <pre>{html.escape(row.baseline.prompt_tail)}</pre>
              <div class='pair'>
                <div class='panel'>
                  <h3>Baseline</h3>
                  <pre>{html.escape(row.baseline.generated_text)}</pre>
                </div>
                <div class='panel'>
                  <h3>Intervention</h3>
                  <pre>{html.escape(row.intervention.generated_text)}</pre>
                </div>
              </div>
            </section>
            """)
    document = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset='utf-8'>
      <title>Story interventions</title>
      <style>
        :root {{ color-scheme: light; }}
        body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 0; background: #f5f7fb; color: #16202a; }}
        main {{ max-width: 1700px; margin: 0 auto; padding: 24px; }}
        h1 {{ margin: 0 0 8px; font-size: 28px; }}
        p {{ margin: 0 0 16px; color: #44515f; }}
        .grid {{ display: grid; gap: 18px; }}
        .card {{ background: white; border: 1px solid #d6deea; border-radius: 14px; padding: 18px; box-shadow: 0 4px 18px rgba(20, 31, 48, 0.06); }}
        .meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
        .meta span {{ background: #eef3ff; color: #2e466e; border-radius: 999px; padding: 4px 10px; font-size: 12px; }}
        h2 {{ margin: 12px 0 8px; font-size: 16px; }}
        h3 {{ margin: 0 0 10px; font-size: 15px; }}
        .pair {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
        .panel {{ background: #fbfcfe; border: 1px solid #e3e8f0; border-radius: 12px; padding: 12px; }}
        pre {{ white-space: pre-wrap; word-break: break-word; background: #f8fafc; border-radius: 10px; padding: 12px; border: 1px solid #e3e8f0; line-height: 1.45; font-size: 13px; margin: 0; }}
        @media (max-width: 1100px) {{ .pair {{ grid-template-columns: 1fr; }} }}
      </style>
    </head>
    <body>
      <main>
        <h1>Story interventions</h1>
        <p>Model: {html.escape(model_path)}. Each story completion was cut at the first <code>&lt;steer&gt;</code> tag and continued with a paired baseline/intervention request.</p>
        <div class='grid'>
          {''.join(cards)}
        </div>
      </main>
    </body>
    </html>
    """
    path.write_text(document)


def main() -> None:
    """Run the story intervention sweep and write `/tmp` artifacts."""
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    story_rows = load_story_rows(path=Path(args.story_results_path))
    cut_points = build_cut_points(
        rows=story_rows,
        intervention_phrase=args.intervention_phrase,
        seed=args.seed,
    )
    prompt_variants = build_prompt_variants(
        cut_points=cut_points,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )
    prompt_batches = build_prompt_batches(prompts=prompt_variants, seed=args.seed)
    result_map = asyncio.run(
        generate_batches_async(
            base_url=args.base_url,
            model_path=args.model_path,
            prompt_batches=prompt_batches,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            async_concurrency=args.async_concurrency,
        )
    )
    paired_rows = pair_results(cut_points=cut_points, result_map=result_map)
    write_jsonl_rows(
        path=Path(args.points_path),
        rows=[asdict(cut_point) for cut_point in cut_points],
    )
    write_jsonl_rows(
        path=Path(args.results_path),
        rows=[row.to_json_dict() for row in paired_rows],
    )
    render_markdown(
        path=Path(args.markdown_path),
        rows=paired_rows,
        model_path=args.model_path,
        base_url=args.base_url,
    )
    render_html(path=Path(args.html_path), rows=paired_rows, model_path=args.model_path)


if __name__ == "__main__":
    main()
