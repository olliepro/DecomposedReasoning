# Bridge Intervention Prompt

## Mode
BRIDGE

## Inputs

- Task id: ${task_id}
- Intervention name: ${intervention_name}
- Required first steer text: ${intervention_variant}
- Exact number of pairs to generate: ${pairs_to_generate_k}
- Steer token limit: ${steer_token_limit}
- Exec token limit: ${exec_token_limit}

## What you are editing

### Prefix up to the bridge point
${trace_prefix}

### Local style window
Use this to match local diction, pace, and specificity.
${local_style_window}

### Next original steer that originally came next
This is the key bridge target. Do not copy it. Do not make it redundant. Write an intervention window that still allows it to feel like a plausible next step.
<steer>${next_original_steer}</steer>

### Optional preview of the next original exec
Use this to avoid solving the same thing twice.
${next_original_exec_preview}

### Tone notes
${notes_on_tone}

## What to do

Generate **exactly ${pairs_to_generate_k} new steer/exec pairs**.

The intervention should introduce the requested reflection, comparison, ranking, critique, synthesis, or local reframing.

It is allowed to:
- slightly change emphasis,
- slow down and inspect assumptions,
- stress-test the current direction,
- narrow to a better immediate next move.

It is **not** allowed to:
- fully replace the trajectory,
- invalidate the whole prefix without justification,
- create a handoff that makes the provided next original steer feel incoherent.

## Compatibility target

When the intervention window ends, ask yourself:

> If the provided next original steer appeared immediately after this window, would it still read as non-redundant, coherent, and useful?

Write toward **yes**.

## Additional instructions

- The **first generated steer** must match this text exactly: `${intervention_variant}`
- Keep every generated steer at or below `${steer_token_limit}` tokens.
- Keep each exec under `${exec_token_limit}` tokens.
- Make the final generated exec naturally set up a return to the old line of reasoning, unless the requested bridge clearly exposes a local issue that would justify later regeneration.

${optional_validation_feedback}
