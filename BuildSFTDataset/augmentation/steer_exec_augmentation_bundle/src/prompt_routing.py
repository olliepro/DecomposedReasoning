from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def choose_prompt_path(
    *, prompts_dir: str | Path, mode: str, intervention_spec: Mapping[str, Any] | None
) -> Path:
    """Return the prompt template path for an intervention mode.

    Args:
        prompts_dir: Directory containing prompt templates.
        mode: Requested editor mode.
        intervention_spec: Optional intervention spec that may override the prompt.

    Returns:
        Prompt template path.

    Example:
        choose_prompt_path(
            prompts_dir="prompts",
            mode="insert",
            intervention_spec={"prompt_template": "non_sequitur_insert.md"},
        )
    """

    if intervention_spec is not None:
        prompt_template = str(intervention_spec.get("prompt_template", "")).strip()
        if prompt_template:
            return Path(prompts_dir) / prompt_template
    default_templates = {
        "insert": "insert.md",
        "bridge": "bridge.md",
        "redirect": "redirect.md",
    }
    return Path(prompts_dir) / default_templates[mode]
