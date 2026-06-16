# Insert Intervention Prompt

## Mode
INSERT

## Inputs

- Task id: ${task_id}
- Intervention name: ${intervention_name}
- Required first steer text: ${intervention_variant}
- Exact number of pairs to generate: ${pairs_to_generate_k}
- Steer token limit: ${steer_token_limit}
- Exec token limit: ${exec_token_limit}

## What you are editing

### Prefix up to the insertion point
${trace_prefix}

### Local style window
Use this only to match local voice and granularity.
${local_style_window}

### Next original steer that should appear immediately after your inserted window
Treat this as a compatibility target. Your inserted window should make this next move still feel natural.
<steer>${next_original_steer}</steer>

### Optional preview of the next original exec
This is here only so you avoid stealing or duplicating the work that the next original exec is about to do.
${next_original_exec_preview}

### Tone notes
${notes_on_tone}

## What to do

Generate **exactly ${pairs_to_generate_k} new steer/exec pairs**.

The intervention should:
- clarify, compress, recap, or lightly structure the reasoning,
- improve the trace locally,
- preserve the existing plan,
- hand off cleanly into the provided next original steer.

The intervention should **not**:
- materially change the trajectory,
- consume the work that the next original steer or exec is supposed to do,
- sound like an external comment about the trace.

## Additional instructions

- The **first generated steer** must match this text exactly: `${intervention_variant}`
- Keep every generated steer at or below `${steer_token_limit}` tokens.
- Keep each exec focused and comfortably under `${exec_token_limit}` tokens.
- Make the final generated exec end at a point where the provided next original steer still feels like the natural next move.

${optional_validation_feedback}
