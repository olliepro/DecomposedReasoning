"""Run paired partial-steer continuations as one batched vLLM request.

Example:
    python Analysis/scripts/run_partial_steer_pair_batch.py \
        --model-path /path/to/checkpoint \
        --base-url http://127.0.0.1:8042/v1 \
        --intervention-phrase "Analyze parts"
"""

from __future__ import annotations

import asyncio
import argparse
import html
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests
from transformers import AutoTokenizer

ACTION_PHRASES: tuple[str, ...] = (
    "Analyze parts",
    "Synthesize whole",
    "Compare paths",
    "Weigh pros",
    "Weigh cons",
    "Broaden search",
    "Trace backward",
    "Use analogy",
    "Organize structure",
    "List unknowns",
)


@dataclass(frozen=True)
class SteerPoint:
    """One partial reasoning prefix cut immediately after `<steer>`.

    Inputs:
        sample_id: Stable sample index in the final output set.
        doc_id: Source document id from the eval artifact.
        rollout_index: Rollout index within the document.
        steer_index: Zero-based chosen steer boundary within the rollout.
        steer_count_total: Total steer boundaries found in the rollout.
        target: Ground-truth target string from the source row.
        user_prompt: Rendered user prompt for chat templating.
        partial_assistant_prefix: Assistant prefix ending at `<steer>`.
        partial_prefix_chars: Character count of the partial assistant prefix.
        action_phrase: Intervention phrase appended inside the steer block.

    Outputs:
        Serializable point record used to construct baseline and intervention
        prompts.
    """

    sample_id: int
    doc_id: int
    rollout_index: int
    steer_index: int
    steer_count_total: int
    target: str
    user_prompt: str
    partial_assistant_prefix: str
    partial_prefix_chars: int
    action_phrase: str


@dataclass(frozen=True)
class PromptVariant:
    """One prompt variant ready for batched generation.

    Inputs:
        sample_id: Parent paired sample id.
        variant: Either `baseline` or `intervention`.
        action_phrase: Short intervention text for display.
        prompt: Full text prompt sent to vLLM.
        prompt_token_count: Token count of the prompt.
        assistant_prefix: Assistant-side prefix embedded in the prompt.

    Outputs:
        Metadata used to reconstruct paired comparison records from the batch
        response.
    """

    sample_id: int
    variant: str
    action_phrase: str
    prompt: str
    prompt_token_count: int
    assistant_prefix: str
    prompt_tail: str


@dataclass(frozen=True)
class PromptBatch:
    """One async prompt-list request sent to vLLM.

    Inputs:
        sample_id: Parent sample id for the paired request.
        seed: Deterministic seed shared by prompts in this request.
        prompts: Ordered baseline/intervention variants for one sample.

    Outputs:
        Serializable batch metadata used to reconstruct final paired rows.

    Example:
        PromptBatch(sample_id=0, seed=7, prompts=(baseline, intervention))
    """

    sample_id: int
    seed: int
    prompts: tuple[PromptVariant, ...]


@dataclass(frozen=True)
class VariantResult:
    """One generated continuation for a prompt variant.

    Inputs:
        prompt_token_count: Prompt token count sent to the model.
        finish_reason: vLLM finish reason string.
        stop_reason: Optional stop reason from the API.
        assistant_prefix: Assistant prefix included before generation.
        generated_text: Newly generated continuation text only.

    Outputs:
        Structured result payload for rendering and JSONL serialization.
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
    """Baseline and intervention continuations for one partial prefix.

    Inputs:
        point: Source steer point metadata.
        baseline: Baseline completion continued from the original prefix.
        intervention: Completion continued from the appended steer phrase.

    Outputs:
        JSON-serializable record for markdown, HTML, and notebook views.
    """

    point: SteerPoint
    baseline: VariantResult
    intervention: VariantResult

    def to_json_dict(self) -> dict[str, Any]:
        """Return the canonical serialized representation for one paired row."""
        data = asdict(self.point)
        data["baseline"] = asdict(self.baseline)
        data["intervention"] = asdict(self.intervention)
        return data


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the batch runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--samples-path",
        default=(
            "Eval/output/manual_20260417/"
            "samples_aime24_olmo3_7b_think_sft_to_think_merged_2213_"
            "checkpoint-432_tp2_avg32_32k_steerprefix_preemptquad_"
            "2026-04-17T16-13-40Z.jsonl"
        ),
    )
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-prefix-chars", type=int, default=600)
    parser.add_argument("--max-prefix-chars", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--async-concurrency", type=int, default=4)
    parser.add_argument("--intervention-phrase", default=None)
    parser.add_argument("--points-path", default="/tmp/olmo_partial_steer_points.jsonl")
    parser.add_argument(
        "--results-path",
        default="/tmp/olmo_partial_steer_continuations.jsonl",
    )
    parser.add_argument("--markdown-path", default="/tmp/olmo_partial_steer_view.md")
    parser.add_argument("--html-path", default="/tmp/olmo_partial_steer_view.html")
    parser.add_argument(
        "--notebook-path",
        default="/tmp/olmo_partial_steer_workflow.ipynb",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def join_rollout_text(raw_rollout: Any) -> str:
    """Return one rollout as a single string.

    Inputs:
        raw_rollout: Either a string rollout or a list of text chunks.

    Outputs:
        Joined rollout text.
    """
    if isinstance(raw_rollout, str):
        return raw_rollout
    assert isinstance(
        raw_rollout, list
    ), f"unexpected rollout type: {type(raw_rollout)}"
    return "".join(str(part) for part in raw_rollout)


def rollout_texts(row: dict[str, Any]) -> list[str]:
    """Return normalized rollout texts from one source row."""
    rollouts = row.get("filtered_resps") or row["resps"]
    return [join_rollout_text(raw_rollout=item) for item in rollouts]


def problem_text(row: dict[str, Any]) -> str:
    """Return the source problem statement for prompt construction."""
    doc = row.get("doc")
    if isinstance(doc, dict):
        return str(doc["Problem"])
    assert isinstance(doc, str), f"unexpected doc type: {type(doc)}"
    return doc


def steer_open_positions(text: str) -> list[int]:
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


def candidate_points(
    rows: Sequence[dict[str, Any]],
    min_prefix_chars: int,
    max_prefix_chars: int,
) -> list[SteerPoint]:
    """Enumerate all eligible partial prefixes across the sample file.

    Example:
        rows = load_jsonl(Path("samples.jsonl"))
        points = candidate_points(rows=rows, min_prefix_chars=600, max_prefix_chars=12000)
    """
    candidates: list[SteerPoint] = []
    for row in rows:
        user_prompt = f"Question: {problem_text(row=row)}\nAnswer:"
        for rollout_index, rollout_text in enumerate(rollout_texts(row=row)):
            positions = steer_open_positions(text=rollout_text)
            for steer_index, cut_index in enumerate(positions):
                prefix = rollout_text[:cut_index]
                if not (min_prefix_chars <= len(prefix) <= max_prefix_chars):
                    continue
                candidates.append(
                    SteerPoint(
                        sample_id=-1,
                        doc_id=int(row["doc_id"]),
                        rollout_index=rollout_index,
                        steer_index=steer_index,
                        steer_count_total=len(positions),
                        target=str(row["target"]),
                        user_prompt=user_prompt,
                        partial_assistant_prefix=prefix,
                        partial_prefix_chars=len(prefix),
                        action_phrase="",
                    )
                )
    return candidates


def select_points(
    candidates: Sequence[SteerPoint],
    sample_count: int,
    seed: int,
    intervention_phrase: str | None,
) -> list[SteerPoint]:
    """Select a deterministic subset of steer points and assign action phrases."""
    assert len(candidates) >= sample_count, "not enough steer points after filtering"
    if intervention_phrase is None:
        assert sample_count <= len(ACTION_PHRASES), "not enough default action phrases"
    chooser = random.Random(seed)
    selected = chooser.sample(list(candidates), sample_count)
    action_phrases = (
        [intervention_phrase] * sample_count
        if intervention_phrase is not None
        else list(ACTION_PHRASES[:sample_count])
    )
    points: list[SteerPoint] = []
    for sample_id, (point, action_phrase) in enumerate(zip(selected, action_phrases)):
        points.append(
            SteerPoint(
                sample_id=sample_id,
                doc_id=point.doc_id,
                rollout_index=point.rollout_index,
                steer_index=point.steer_index,
                steer_count_total=point.steer_count_total,
                target=point.target,
                user_prompt=point.user_prompt,
                partial_assistant_prefix=point.partial_assistant_prefix,
                partial_prefix_chars=point.partial_prefix_chars,
                action_phrase=action_phrase,
            )
        )
    return points


def write_points(path: Path, points: Iterable[SteerPoint]) -> None:
    """Write steer points to JSONL for reproducibility."""
    with path.open("w") as handle:
        for point in points:
            handle.write(json.dumps(asdict(point)))
            handle.write("\n")


def build_base_prompt(tokenizer: Any, user_prompt: str) -> str:
    """Return the chat-templated user prompt with an open assistant turn."""
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
    ).rstrip()


def attach_assistant_prefix(base_prompt: str, assistant_prefix: str) -> str:
    """Append the assistant prefix to the chat template safely.

    Inputs:
        base_prompt: Chat-template output ending at an open assistant turn.
        assistant_prefix: Saved assistant prefix from the sampled rollout.

    Outputs:
        Full completion prompt ready for generation.

    Example:
        attach_assistant_prefix("...assistant\\n<think>", "<think><steer>")
        attach_assistant_prefix("...assistant", "<think><steer>")
    """
    if base_prompt.endswith("<think>") and assistant_prefix.startswith("<think>"):
        return base_prompt + assistant_prefix[len("<think>") :]
    return base_prompt + assistant_prefix


def prompt_variants(
    points: Sequence[SteerPoint],
    tokenizer: Any,
    max_new_tokens: int,
) -> list[PromptVariant]:
    """Build baseline and intervention prompts for all selected points."""
    variants: list[PromptVariant] = []
    for point in points:
        base_prompt = build_base_prompt(
            tokenizer=tokenizer,
            user_prompt=point.user_prompt,
        )
        prefixes = (
            ("baseline", point.partial_assistant_prefix),
            (
                "intervention",
                point.partial_assistant_prefix + point.action_phrase,
            ),
        )
        for variant_name, assistant_prefix in prefixes:
            prompt = attach_assistant_prefix(
                base_prompt=base_prompt,
                assistant_prefix=assistant_prefix,
            )
            prompt_token_count = len(
                tokenizer(prompt, add_special_tokens=False)["input_ids"]
            )
            assert prompt_token_count + max_new_tokens <= 32768, prompt_token_count
            variants.append(
                PromptVariant(
                    sample_id=point.sample_id,
                    variant=variant_name,
                    action_phrase=point.action_phrase,
                    prompt=prompt,
                    prompt_token_count=prompt_token_count,
                    assistant_prefix=assistant_prefix,
                    prompt_tail=prompt[-500:],
                )
            )
    return variants


def prompt_batches(
    prompts: Sequence[PromptVariant],
    seed: int,
) -> list[PromptBatch]:
    """Group flat prompt variants into paired async prompt-list requests.

    Inputs:
        prompts: Flat baseline/intervention prompt list.
        seed: Base seed for deterministic per-sample request seeds.

    Outputs:
        Ordered prompt batches, one per paired sample.
    """
    grouped: dict[int, list[PromptVariant]] = {}
    for prompt in prompts:
        grouped.setdefault(prompt.sample_id, []).append(prompt)
    batches: list[PromptBatch] = []
    for sample_id in sorted(grouped):
        sample_prompts = tuple(
            sorted(grouped[sample_id], key=lambda item: item.variant)
        )
        assert len(sample_prompts) == 2, "expected baseline/intervention pair"
        batches.append(
            PromptBatch(
                sample_id=sample_id,
                seed=seed + sample_id,
                prompts=sample_prompts,
            )
        )
    return batches


def generate_batch_results(
    *,
    base_url: str,
    model_path: str,
    prompt_batch: PromptBatch,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[VariantResult]:
    """Generate one paired batch in a single prompt-list API call.

    Inputs:
        base_url: Base vLLM OpenAI-compatible API url.
        model_path: Served model id/path.
        prompt_batch: One paired prompt batch for a single sample id.

    Outputs:
        Ordered variant results matching the paired prompt order.
    """
    payload = {
        "model": model_path,
        "prompt": [prompt.prompt for prompt in prompt_batch.prompts],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
        "seed": prompt_batch.seed,
    }
    response = requests.post(
        url=f"{base_url}/completions",
        json=payload,
        timeout=3600,
    )
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


async def batched_generate_async(
    *,
    base_url: str,
    model_path: str,
    prompt_batches: Sequence[PromptBatch],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    async_concurrency: int,
) -> dict[tuple[int, str], VariantResult]:
    """Generate paired continuations concurrently across async prompt batches.

    Inputs:
        prompt_batches: Paired prompt batches, one request per sample.
        async_concurrency: Max simultaneous vLLM requests.

    Outputs:
        Mapping keyed by `(sample_id, variant)` for stable reconstruction.

    Example:
        results = asyncio.run(
            batched_generate_async(
                base_url="http://127.0.0.1:8042/v1",
                model_path="/checkpoint",
                prompt_batches=batches,
                max_new_tokens=512,
                temperature=0.6,
                top_p=0.95,
                async_concurrency=4,
            )
        )
    """
    semaphore = asyncio.Semaphore(async_concurrency)
    result_map: dict[tuple[int, str], VariantResult] = {}

    async def run_one(prompt_batch: PromptBatch) -> None:
        async with semaphore:
            batch_results = await asyncio.to_thread(
                generate_batch_results,
                base_url=base_url,
                model_path=model_path,
                prompt_batch=prompt_batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        for prompt, result in zip(prompt_batch.prompts, batch_results):
            result_map[(prompt.sample_id, prompt.variant)] = result

    await asyncio.gather(
        *(run_one(prompt_batch=prompt_batch) for prompt_batch in prompt_batches)
    )
    return result_map


def pair_results(
    points: Sequence[SteerPoint],
    result_map: dict[tuple[int, str], VariantResult],
) -> list[PairedResult]:
    """Group flat prompt results back into paired sample rows."""
    paired: list[PairedResult] = []
    for point in points:
        paired.append(
            PairedResult(
                point=point,
                baseline=result_map[(point.sample_id, "baseline")],
                intervention=result_map[(point.sample_id, "intervention")],
            )
        )
    return paired


def preview_text(text: str, limit: int = 160) -> str:
    """Return a compact preview snippet for markdown tables."""
    return " ".join(text.split())[:limit]


def write_results(path: Path, rows: Iterable[PairedResult]) -> None:
    """Write paired result rows to JSONL."""
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json_dict()))
            handle.write("\n")


def render_markdown(
    path: Path, rows: Sequence[PairedResult], model_path: str, base_url: str
) -> None:
    """Render a concise markdown summary of the paired comparison."""
    lines = [
        "# Partial-steer continuations",
        "",
        "Mode: `async paired prompt batches`  ",
        f"Model: `{model_path}`  ",
        f"Base URL: `{base_url}`  ",
        "",
        "| id | doc | rollout | steer | phrase | base toks | base finish | intervention toks | intervention finish |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {sample_id} | {doc_id} | {rollout_index} | {steer_index} | {action_phrase} | {base_toks} | {base_finish} | {intervention_toks} | {intervention_finish} |".format(
                sample_id=row.point.sample_id,
                doc_id=row.point.doc_id,
                rollout_index=row.point.rollout_index,
                steer_index=row.point.steer_index,
                action_phrase=row.point.action_phrase.replace("|", "\\|"),
                base_toks=row.baseline.prompt_token_count,
                base_finish=row.baseline.finish_reason,
                intervention_toks=row.intervention.prompt_token_count,
                intervention_finish=row.intervention.finish_reason,
            )
        )
        lines.append("")
        lines.append(f"Baseline preview: `{row.baseline.continuation_preview}`")
        lines.append("")
        lines.append(f"Intervention preview: `{row.intervention.continuation_preview}`")
        lines.append("")
    path.write_text("\n".join(lines))


def render_html(path: Path, rows: Sequence[PairedResult], model_path: str) -> None:
    """Render the side-by-side HTML comparison view."""
    cards: list[str] = []
    for row in rows:
        cards.append(
            f"""
            <section class='card'>
              <div class='meta'>
                <span>sample {row.point.sample_id}</span>
                <span>doc {row.point.doc_id}</span>
                <span>rollout {row.point.rollout_index}</span>
                <span>steer {row.point.steer_index}/{row.point.steer_count_total}</span>
                <span>phrase: {html.escape(row.point.action_phrase)}</span>
              </div>
              <h2>Prompt tail</h2>
              <pre>{html.escape(row.baseline.prompt_tail)}</pre>
              <div class='pair'>
                <div class='panel'>
                  <h3>Baseline</h3>
                  <div class='submeta'>
                    <span>{row.baseline.prompt_token_count} prompt toks</span>
                    <span>{html.escape(str(row.baseline.finish_reason))}</span>
                  </div>
                  <pre>{html.escape(row.baseline.generated_text)}</pre>
                </div>
                <div class='panel'>
                  <h3>Intervention</h3>
                  <div class='submeta'>
                    <span>{row.intervention.prompt_token_count} prompt toks</span>
                    <span>{html.escape(str(row.intervention.finish_reason))}</span>
                  </div>
                  <pre>{html.escape(row.intervention.generated_text)}</pre>
                </div>
              </div>
            </section>
            """
        )
    document = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset='utf-8'>
      <title>Batched Baseline vs Intervention</title>
      <style>
        :root {{ color-scheme: light; }}
        body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 0; background: #f5f7fb; color: #16202a; }}
        main {{ max-width: 1700px; margin: 0 auto; padding: 24px; }}
        h1 {{ margin: 0 0 8px; font-size: 28px; }}
        p {{ margin: 0 0 20px; color: #44515f; }}
        .grid {{ display: grid; gap: 18px; }}
        .card {{ background: white; border: 1px solid #d6deea; border-radius: 14px; padding: 18px; box-shadow: 0 4px 18px rgba(20, 31, 48, 0.06); }}
        .meta, .submeta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
        .meta span {{ background: #eef3ff; color: #2e466e; border-radius: 999px; padding: 4px 10px; font-size: 12px; }}
        .submeta span {{ background: #f1f5f9; color: #415466; border-radius: 999px; padding: 3px 9px; font-size: 12px; }}
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
        <h1>Batched Baseline vs Intervention</h1>
        <p>Model: {html.escape(model_path)}. Paired prompt-list requests were sent concurrently.</p>
        <div class='grid'>
          {''.join(cards)}
        </div>
      </main>
    </body>
    </html>
    """
    path.write_text(document)


def render_notebook(path: Path, results_path: Path, html_path: Path) -> None:
    """Render a lightweight notebook describing the async batched workflow."""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Batched partial-steer workflow\n",
                    "This notebook reads the paired comparison artifacts generated from concurrent prompt-list completion batches.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import json\n",
                    "import pandas as pd\n",
                    f"RESULTS_PATH = Path('{results_path}')\n",
                    "rows = [json.loads(line) for line in RESULTS_PATH.open()]\n",
                    "df = pd.DataFrame([\n",
                    "    {\n",
                    "        'sample_id': row['sample_id'],\n",
                    "        'action_phrase': row['action_phrase'],\n",
                    "        'baseline_toks': row['baseline']['prompt_token_count'],\n",
                    "        'baseline_finish': row['baseline']['finish_reason'],\n",
                    "        'intervention_toks': row['intervention']['prompt_token_count'],\n",
                    "        'intervention_finish': row['intervention']['finish_reason'],\n",
                    "    }\n",
                    "    for row in rows\n",
                    "])\n",
                    "df\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "row = rows[0]\n",
                    "print('PROMPT TAIL')\n",
                    "print(row['baseline']['prompt_tail'])\n",
                    "print()\n",
                    "print('BASELINE')\n",
                    "print(row['baseline']['generated_text'])\n",
                    "print()\n",
                    "print('INTERVENTION')\n",
                    "print(row['intervention']['generated_text'])\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"Rendered HTML: `{html_path}`\n"],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.9"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2))


def main() -> None:
    """Run the full batched partial-steer workflow and write `/tmp` artifacts."""
    args = parse_args()
    rows = load_jsonl(path=Path(args.samples_path))
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    candidates = candidate_points(
        rows=rows,
        min_prefix_chars=args.min_prefix_chars,
        max_prefix_chars=args.max_prefix_chars,
    )
    points = select_points(
        candidates=candidates,
        sample_count=args.sample_count,
        seed=args.seed,
        intervention_phrase=args.intervention_phrase,
    )
    prompts = prompt_variants(
        points=points,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )
    prompt_batch_list = prompt_batches(prompts=prompts, seed=args.seed)
    result_map = asyncio.run(
        batched_generate_async(
            base_url=args.base_url,
            model_path=args.model_path,
            prompt_batches=prompt_batch_list,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            async_concurrency=args.async_concurrency,
        )
    )
    paired = pair_results(points=points, result_map=result_map)
    write_points(path=Path(args.points_path), points=points)
    write_results(path=Path(args.results_path), rows=paired)
    render_markdown(
        path=Path(args.markdown_path),
        rows=paired,
        model_path=args.model_path,
        base_url=args.base_url,
    )
    render_html(path=Path(args.html_path), rows=paired, model_path=args.model_path)
    render_notebook(
        path=Path(args.notebook_path),
        results_path=Path(args.results_path),
        html_path=Path(args.html_path),
    )


if __name__ == "__main__":
    main()
