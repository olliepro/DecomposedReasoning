# Non-Sequitur Insert Intervention Prompt

## Mode
INSERT_NON_SEQUITUR

## Inputs

- Task id: ${task_id}
- Intervention name: ${intervention_name}
- Reference steer idea: ${intervention_variant}
- Exact number of pairs to generate: ${pairs_to_generate_k}
- Steer token limit: ${steer_token_limit}
- Exec token limit: ${exec_token_limit}

## What you are editing

### Prefix up to the insertion point
${trace_prefix}

### Local style window
Use this only to match local voice and granularity.
${local_style_window}

### Tone notes
${notes_on_tone}

## What to do

Generate **exactly ${pairs_to_generate_k} new steer/exec pairs**.

This intervention is an intentional **non-sequitur detour**.
It may be:
- loosely related to the surrounding task,
- orthogonal to it,
- or mildly adversarial to the current line of thought.

The intervention should:
- be self-contained and locally coherent,
- keep each exec tightly aligned to its steer,
- read like something the same internal reasoner might suddenly do,
- stay concrete rather than generic,
- allow for odd, playful, or slightly offbeat local moves,
- favor a bit of internal texture or local complication over the most trivial possible execution,
- and, when the steer depends on the visible prefix, anchor the exec to that local context instead of inventing filler.

The intervention should **not**:
- say that it is irrelevant, inserted, random, weird, or adversarial,
- refer to a “detour” or “non-sequitur,”
- try to hand off into the next original steer,
- summarize the surrounding task unless the chosen steer explicitly asks for it,
- break the strict `<steer>/<exec>` alternation.

## Additional instructions

- Use `${intervention_variant}` as a reference idea, not an exact string to copy.
- Keep the first steer short and natural; variation across runs is good.
- Keep every generated steer at or below `${steer_token_limit}` tokens.
- Keep each exec focused and comfortably under `${exec_token_limit}` tokens.
- If the steer asks for a tiny computation or check, actually carry it out in the exec.
- If the steer is context dependent, make the exec explicitly reference the nearby reasoning state.
- If later generated steers appear, keep them short, directive, and natural for the detour.

${optional_validation_feedback}

DO NOT TRY TO SOLVE THE PROBLEM... YOUR GOAL IS TO MAKE A NON-SEQUITUR
