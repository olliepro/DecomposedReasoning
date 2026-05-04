# Bridge Judge Prompt

You are deciding whether a newly inserted bridge intervention window can still be followed by the original next steer **without sounding awkward, redundant, or off-trajectory**.

## Inputs

### Prefix before insertion
${trace_prefix}

### Inserted intervention window
${inserted_window}

### Original next steer
<steer>${next_original_steer}</steer>

### Optional preview of the next original exec
${next_original_exec_preview}

## Decision rule

Return `keep_suffix` only if the original next steer would still feel:

- coherent after the inserted window,
- non-redundant,
- goal-directed,
- appropriately scoped.

Return `regenerate_suffix` if the inserted window makes the original next steer feel:

- repetitive,
- too abrupt,
- logically mismatched,
- or like the reasoning should now continue differently.

Be conservative. If uncertain, prefer `regenerate_suffix`.
