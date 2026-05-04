# Steer/Exec Intervention Augmentation Guide

This bundle is for augmenting reasoning traces written as alternating `<steer>` and `<exec>` blocks.

The core idea is simple:

1. Cut an existing trace **at a pair boundary**.
2. Ask a stronger **intervention model** to generate only a **short intervention window**.
3. Splice that window into the trace.
4. Either keep the original suffix, or regenerate from the augmented prefix later with the original model.

The goal is **not** to make every augmentation feel like a full rewrite. Most natural interventions are small and local. Only some should noticeably alter the downstream reasoning trajectory.

---

## Why this workflow is useful

If the original model already produces good reasoning traces, but does **not** reliably handle mid-generation metacognitive interventions, then a clean way to teach the behavior is:

- preserve the original trace structure,
- insert a plausible intervention at realistic positions,
- use the intervention model only for the new local span,
- let the original model learn from these edited traces during fine-tuning.

That avoids over-relying on a different teacher for the whole trace.

---

## What “natural-feeling” means here

A good augmentation should read like something the same internal reasoner *could plausibly have said next*.

A bad augmentation tends to have one of these failure modes:

- it sounds externally edited rather than internally motivated,
- it becomes too generic and stops advancing the object-level solution,
- it duplicates what the next original block was already going to do,
- it changes direction but then incorrectly keeps the old suffix,
- it is structurally malformed, for example two `<exec>` blocks in a row.

This bundle is organized to reduce those failure modes.

---

## Core terms

### Pair boundary
A cut point between complete `<steer>/<exec>` pairs.
Always cut here. Do **not** cut in the middle of an `<exec>` block.

### Prefix
Everything before the intervention window.

### Intervention window
The newly generated blocks inserted by the intervention model.

### Next original steer
The `<steer>` block that originally came right after the cut point.

This is especially important for **insert** and **bridge** interventions, because it tells the intervention model what the old trace was about to do.

### Post-splice policy
What to do with the original suffix after you insert the new window.

---

## The three augmentation modes

### Insert
Use when the intervention is mostly a recap, clarification, explicit constraint check, or light structuring move.

What happens:
- the intervention model writes a short window,
- you keep the original suffix,
- the new window should make the next original steer feel **more natural**, not less.

Typical size:
- usually **1 pair**
- occasionally **2 pairs** if the intervention needs a setup and a return

### Bridge
Use when the intervention introduces some critique, ranking, comparison, or local reframing, but you still want the old next move to remain plausible.

What happens:
- the intervention model writes a short window,
- you show it the **next original steer** as a compatibility target,
- after insertion, you either keep the old suffix or decide it no longer fits.

Typical size:
- usually **1 to 2 pairs**
- occasionally **3** for more substantial critiques

### Redirect
Use when the intervention should intentionally alter the trajectory.

Examples:
- backtrack
- simplify
- work backwards
- switch abstraction
- create a toy problem first

What happens:
- the intervention model writes a short redirect window,
- the old suffix is treated as historical context only,
- you regenerate from the new augmented prefix later with the original model.

Typical size:
- usually **2 to 4 pairs**

---

## Why insert and bridge prompts include the next original steer

This is the most important prompt-design choice in the bundle.

If you do **not** show the intervention model the next original steer, it will often produce one of two bad outcomes:

- an intervention that is locally sensible but makes the next original steer redundant or awkward,
- an intervention that solves too much, leaving the suffix with nothing meaningful left to do.

So for **insert** and **bridge** prompts, the intervention model sees:

- the prefix,
- a small local style window,
- the exact intervention variant to express,
- the next original steer that should come after the intervention,
- and optionally a short preview of the next original exec.

That extra context makes “slotting in” much easier.

For **redirect** prompts, the next original steer is shown only as context for what the old trace was about to do, not as something to preserve.

---

## Recommended default recipe

Start with these defaults:

- one intervention per trace
- mode mix: **60% insert / 25% bridge / 15% redirect**
- sample cut points at **pair boundaries**
- validate every input trace before augmentation
- validate every generated intervention before splicing
- keep `<exec>` blocks under **512 tokens**
- use a bridge judge for `keep_if_next_steer_still_fits_else_regenerate`

These defaults are conservative. They optimize for usable training data over maximum novelty.

---

## Expected input format

The example script accepts JSONL records.

Preferred shape:

```json
{
  "task_id": "ex-001",
  "user_prompt": "Prove that the sum of two odd integers is even.",
  "trace": "<steer>restate the claim</steer><exec>We want to show...</exec>...",
  "answer": "Therefore the sum is even."
}
```

It also tolerates:

- `trace_blocks`: a list like `[{ "type": "steer", "text": "..." }, ...]`
- `assistant`: a full assistant message that contains a `<think>...</think>` region

The validator expects the extracted reasoning trace to alternate:

`<steer>, <exec>, <steer>, <exec>, ...`

---

## Validation rules

The example pipeline validates **both** the existing trace and the generated intervention window.

### Required structural checks
- only `steer` and `exec` blocks are allowed
- blocks must strictly alternate
- the first block must be `steer`
- block count must be even

### Intervention-window checks
- exact number of pairs requested
- first new steer should match the chosen intervention variant exactly
- every generated `<exec>` must be below the token limit
- no empty block text

### Why the exec-token limit matters
Your augmented traces become training targets. If intervention exec blocks are too long, they stop acting like “behavioral nudges” and start acting like full rewrites. A hard cap keeps the behavior local.

The example code lets you choose the tokenizer used for this check. For best results, use the tokenizer of the model you care most about when enforcing the limit.

---
## Async execution model

The batch runner now issues OpenRouter calls asynchronously and reuses a single pooled HTTP client.

Why this matters:

- it lets multiple augmentation jobs stay in flight at once,
- it avoids paying connection setup costs on every request,
- and it gives you one place to cap request pressure with `--max-concurrency`.

A good starting point is `--max-concurrency 4` to `8`, then increase only if your provider routing and rate limits are stable.

---

## Suggested cut-point strategy

Do **not** choose cut points uniformly over raw tokens.

Instead:

1. parse the trace into complete pairs,
2. choose an intervention,
3. use that intervention's `preferred_slots`,
4. sample a cut point from the matching region.

A good practical mapping is:

- **early**: first 25% of available pair boundaries
- **mid**: middle 50%
- **late**: last 25%

If a trace is too short to support the preferred slot cleanly, fall back to any valid interior cut point.

---

## Choosing pair counts

Do not ask every intervention to write 1 to 10 pairs.

That makes many traces feel inflated or unnatural.

A better heuristic is:

- **local compressive** interventions: 1 pair
- **diagnostic** interventions: 1 to 2 pairs
- **redirective** interventions: 2 to 4 pairs

The `interventions.json` file already includes `pairs_to_generate` and a `default` value for each intervention.

---

## What the prompts are doing

The prompt files in `prompts/` are designed so that the intervention model generates **only the intervention window**, not the whole suffix.

That has two benefits:

- it keeps the intervention model focused on the local edit,
- it lets you use the original model later to regenerate from the new prefix when needed.

The prompt files are written in Markdown so they are easy to inspect and revise during prompt engineering. The batch script and the notebook both read the same files, so your prompt iterations transfer directly into the pipeline.

---

## Structured output schema

The OpenRouter call uses the chat completions API with a **JSON Schema** response format. The intervention model returns a small object like:

```json
{
  "blocks": [
    { "type": "steer", "text": "..." },
    { "type": "exec", "text": "..." }
  ]
}
```

The API enforces the outer structure; the local validator enforces the alternation pattern and token limit.

This split is deliberate:
- schema handles shape,
- local code handles semantic constraints that are easier to validate outside the model.

---

## Output records

The example script writes augmented JSONL records with both the edited trace and metadata.

Typical fields include:

- original record fields
- `augmentation` metadata:
  - chosen intervention
  - chosen variant
  - mode
  - cut position
  - suffix decision
  - validation diagnostics
- `augmented_prefix_trace`
- `augmented_full_trace` when the suffix is kept
- `regen_seed` so you can continue from the augmented prefix later

The output is designed to support either:
- direct supervised fine-tuning on the spliced trace, or
- a second-stage regeneration pass with the original model.

---

## How the bridge judge works

Bridge interventions are the ambiguous case.

They are meant to pressure-test or reframe the reasoning **without necessarily** replacing the old suffix.

So the bundle includes an optional bridge-judge prompt that asks a second structured-output question:

> If the original next steer appeared immediately after the inserted window, would the trace still feel coherent, non-redundant, and goal-directed?

The judge returns one of:
- `keep_suffix`
- `regenerate_suffix`

Use it when you want a more disciplined separation between bridge and redirect behavior.

---

## A good starting workflow

### 1. Validate the raw dataset
Run the validator and see how many traces are already malformed or have oversized exec blocks.

### 2. Start with a narrow intervention subset
A good first batch is:
- restate task and approach
- state known constraints
- summarize progress
- check for contradictions
- stop and list alternate approaches
- start with a simpler stepping stone

These cover recap, local critique, and redirective behavior without being too exotic.

### 3. Use the notebook first
Use `notebooks/prompt_engineering.ipynb` on a few examples before launching a large batch.

### 4. Run small batches
Inspect 20–50 augmented outputs manually before scaling up.

### 5. Track rejection reasons
Common rejection reasons usually tell you whether:
- the prompts are underspecified,
- the token limit is too tight,
- or a specific intervention should move to redirect mode more often.

---

## Practical defaults in the example code

The example implementation uses:

- OpenRouter model: `openai/gpt-oss-20b`
- async HTTP calls via `httpx.AsyncClient`
- bounded concurrency in the batch runner
- `response_format.type = "json_schema"`
- `provider.require_parameters = true`
- optional `provider.data_collection = "deny"`
- optional `response-healing` plugin
- a `regen_seed` object rather than forcing continuation to completion

That keeps the augmentation step narrowly scoped.

---

## Working with vLLM later

This bundle does **not** force a full continuation pass.

Instead, it emits a `regen_seed` object containing the original user prompt and the augmented assistant-side reasoning prefix. That is the handoff point for your original model.

Because vLLM exposes an OpenAI-compatible server, you can later decide whether to:
- continue via a chat-style request with assistant prefill if your serving stack supports it cleanly,
- or convert the augmented prefix into a completion prompt that matches your model's training template.

The bundle leaves that choice to you because continuation behavior can vary across models and chat templates.

---

## File layout

```text
steer_exec_augmentation_bundle/
├── guide.md
├── interventions.json
├── requirements.txt
├── examples/
│   └── sample_traces.jsonl
├── prompts/
│   ├── system.md
│   ├── insert.md
│   ├── bridge.md
│   ├── redirect.md
│   ├── bridge_judge.md
│   └── judge_system.md
├── scripts/
│   └── augment_traces.py
├── src/
│   ├── __init__.py
│   └── trace_augmentor.py
└── notebooks/
    └── prompt_engineering.ipynb
```

---

## Example command

```bash
export OPENROUTER_API_KEY=...
python scripts/augment_traces.py   --input examples/sample_traces.jsonl   --output out/augmented.jsonl   --interventions interventions.json   --prompts-dir prompts   --openrouter-model openai/gpt-oss-20b   --augmentations-per-record 1   --max-concurrency 8   --run-bridge-judge   --wrap-think
```

For offline pipeline testing without hitting the API:

```bash
python scripts/augment_traces.py   --input examples/sample_traces.jsonl   --output out/mock_augmented.jsonl   --interventions interventions.json   --prompts-dir prompts   --mock-intervention   --wrap-think
```

---

## Notes on API assumptions

The example code is written against the current public docs for:

- OpenRouter structured outputs in the chat completions API
- OpenRouter provider routing with `require_parameters`
- OpenRouter data-policy filtering with `data_collection`
- OpenRouter's `response-healing` plugin
- vLLM's OpenAI-compatible server

Those assumptions can change over time. If you upgrade your routing or serving stack, re-check the docs before doing a large run.

References:
- OpenRouter structured outputs: https://openrouter.ai/docs/guides/features/structured-outputs
- OpenRouter API reference: https://openrouter.ai/docs/api/reference/overview
- OpenRouter provider routing: https://openrouter.ai/docs/guides/routing/provider-selection
- OpenRouter model page for gpt-oss-20b: https://openrouter.ai/openai/gpt-oss-20b
- vLLM OpenAI-compatible server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
