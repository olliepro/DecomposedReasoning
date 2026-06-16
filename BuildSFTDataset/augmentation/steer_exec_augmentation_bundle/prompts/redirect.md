# Redirect Intervention Prompt

## Mode
REDIRECT

## Inputs

- Task id: ${task_id}
- Intervention name: ${intervention_name}
- Required first steer text: ${intervention_variant}
- Exact number of pairs to generate: ${pairs_to_generate_k}
- Steer token limit: ${steer_token_limit}
- Exec token limit: ${exec_token_limit}

## What you are editing

### Prefix up to the redirect point
${trace_prefix}

### Local style window
Use this to match the voice of the existing trace while still changing direction.
${local_style_window}

### Original next steer for context only
This shows what the old trace was about to do. You are **not** trying to preserve it.
<steer>${next_original_steer}</steer>

### Optional preview of the next original exec
Use this only to understand what the old continuation would have tried to do.
${next_original_exec_preview}

### Tone notes
${notes_on_tone}

## What to do

Generate **exactly ${pairs_to_generate_k} new steer/exec pairs**.

This intervention should **change the local trajectory**. Good redirect windows often do one of the following:

- backtrack to the last solid point,
- question the top-level strategy,
- simplify aggressively,
- move to a better abstraction,
- create a toy problem,
- work backwards,
- choose a different family of next steps.

The new window should make a later regeneration step from the augmented prefix feel justified.

## Additional instructions

- The **first generated steer** must match this text exactly: `${intervention_variant}`
- Keep every generated steer at or below `${steer_token_limit}` tokens.
- Keep each exec under `${exec_token_limit}` tokens.
- Do not try to rejoin the old suffix.
- End in a way that clearly motivates a *different* next move than the one the old trace was about to take.

${optional_validation_feedback}
