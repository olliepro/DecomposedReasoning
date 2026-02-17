"""Typed objects used by the steer-branching analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ApiMode = Literal["completions", "chat"]
SelectionPolicy = Literal["random"]
ContentFormat = Literal["string", "openai"]


@dataclass(frozen=True)
class ApiModeConfig:
    """API mode and capability guards.

    Args:
        default_mode: Primary endpoint mode used for requests.
        allow_fallback: Enables fallback from chat to completions mode.
        max_server_logprobs: Upper bound enforced for requested top logprobs.

    Returns:
        Dataclass storing API behavior controls.
    """

    default_mode: ApiMode = "completions"
    allow_fallback: bool = True
    max_server_logprobs: int = 20

    def capped_top_logprobs(self, *, requested_top_logprobs: int) -> int:
        """Cap requested top-logprob count to server limits.

        Args:
            requested_top_logprobs: Requested number of alternatives per token.

        Returns:
            Effective top-logprob count after capping.
        """
        assert requested_top_logprobs >= 0, "requested_top_logprobs must be >= 0"
        return min(requested_top_logprobs, self.max_server_logprobs)


@dataclass(frozen=True)
class TemplateConfig:
    """Prompt-template controls for chat and completions request modes.

    Args:
        use_raw_im_template: Uses raw `<|im_start|>` prompt strings in completions.
        add_generation_prompt: Requests server-side generation prompt for chat mode.
        continue_final_message: Continue final assistant message in chat mode.
        chat_template: Optional explicit template string.
        chat_template_kwargs: Optional server template kwargs.
        content_format: Chat content format (`string` or `openai`).

    Returns:
        Dataclass storing prompt templating controls.
    """

    use_raw_im_template: bool = True
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    chat_template: str | None = None
    chat_template_kwargs: dict[str, object] = field(default_factory=dict)
    content_format: ContentFormat = "string"

    def validate(self) -> None:
        """Validate mutually exclusive chat templating options.

        Args:
            None.

        Returns:
            None.
        """
        invalid_combo = self.add_generation_prompt and self.continue_final_message
        assert (
            not invalid_combo
        ), "add_generation_prompt and continue_final_message conflict"


@dataclass(frozen=True)
class RunConfig:
    """Runtime configuration for one steer-branching run.

    Args:
        base_url: Base OpenAI-compatible URL (`.../v1`).
        model: Model identifier passed to vLLM.
        prompt: User prompt content.
        output_root: Directory where `run_id` output is written.
        api_mode_config: Endpoint mode and capability settings.
        template_config: Prompt-template behavior.
        temperature: Sampling temperature.
        top_p: Nucleus sampling value.
        max_total_tokens: Aggregate generation budget for run.
        branch_factor: Number of steer candidates sampled per branch step.
        n_keep: Number of branches kept per step.
        selection_policy: Branch selection policy.
        max_steer_tokens: Max tokens sampled per steer candidate.
        max_steps: Max number of branch events.
        top_logprobs: Requested top alternatives per generated token.
        seed: RNG seed for reproducibility.
        rollout_chunk_tokens: Tokens per rollout chunk before rescanning tags.
        boundary_pattern: Regex marking branch boundary.

    Returns:
        Dataclass storing one end-to-end run configuration.

    Example:
        >>> cfg = RunConfig(base_url="http://127.0.0.1:8000/v1", model="m", prompt="p")
        >>> cfg.branch_factor
        100
    """

    base_url: str
    model: str
    prompt: str
    output_root: Path = Path("output")
    api_mode_config: ApiModeConfig = ApiModeConfig()
    template_config: TemplateConfig = TemplateConfig()
    temperature: float = 0.6
    top_p: float = 0.95
    max_total_tokens: int = 32768
    branch_factor: int = 100
    n_keep: int = 1
    selection_policy: SelectionPolicy = "random"
    max_steer_tokens: int = 15
    max_steps: int = 100
    top_logprobs: int = 20
    seed: int = 0
    rollout_chunk_tokens: int = 256
    boundary_pattern: str = r"<steer\b[^>]*>"

    def validate(self) -> None:
        """Validate configuration values.

        Args:
            None.

        Returns:
            None.
        """
        assert self.base_url.strip(), "base_url must be non-empty"
        assert self.model.strip(), "model must be non-empty"
        assert self.prompt.strip(), "prompt must be non-empty"
        assert 0.0 <= self.temperature, "temperature must be >= 0"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
        assert self.max_total_tokens >= 1, "max_total_tokens must be >= 1"
        assert self.branch_factor >= 1, "branch_factor must be >= 1"
        assert self.n_keep == 1, "current implementation requires n_keep == 1"
        assert self.max_steer_tokens >= 1, "max_steer_tokens must be >= 1"
        assert self.max_steps >= 1, "max_steps must be >= 1"
        self.template_config.validate()

    def capped_top_logprobs(self) -> int:
        """Resolve effective top-logprob request.

        Args:
            None.

        Returns:
            Capped top-logprob request.
        """
        return self.api_mode_config.capped_top_logprobs(
            requested_top_logprobs=self.top_logprobs
        )


@dataclass(frozen=True)
class TokenAlternative:
    """One alternative token from top-logprob distribution.

    Args:
        token: Token text.
        logprob: Natural log probability.
        probability: Linear-space probability.

    Returns:
        Dataclass storing one token alternative.
    """

    token: str
    logprob: float
    probability: float

    def as_text(self) -> str:
        """Format token alternative for UI hover displays.

        Args:
            None.

        Returns:
            One-line formatted token entry.
        """
        return (
            f"token={self.token!r}, logprob={self.logprob:.4f}, "
            f"prob={self.probability:.4f}"
        )


@dataclass(frozen=True)
class TokenStat:
    """Per-token probability and entropy statistics.

    Args:
        source: Source label (`rollout` or `candidate`).
        step_index: Branch step index.
        candidate_index: Candidate index for candidate tokens, else -1.
        token_index: Token index within source sequence.
        token: Selected token text.
        logprob: Selected token logprob.
        probability: Selected token probability.
        entropy: Approximate token entropy.
        alternatives: Top alternative tokens for hover display.

    Returns:
        Dataclass storing one token metric record.
    """

    source: str
    step_index: int
    candidate_index: int
    token_index: int
    token: str
    logprob: float
    probability: float
    entropy: float
    alternatives: tuple[TokenAlternative, ...] = ()

    def hover_text(self) -> str:
        """Build canonical hover text for the static report.

        Args:
            None.

        Returns:
            Multiline string for tooltip display.
        """
        lines = [
            f"token={self.token!r}",
            f"logprob={self.logprob:.4f}",
            f"prob={self.probability:.4f}",
            f"entropy={self.entropy:.4f}",
        ]
        alt_lines = [alternative.as_text() for alternative in self.alternatives]
        return "\n".join(lines + (["alternatives:"] + alt_lines if alt_lines else []))


@dataclass(frozen=True)
class SteerCandidate:
    """One steer candidate sampled at a branch event.

    Args:
        step_index: Branch step index.
        candidate_index: Candidate index within sampled batch.
        text: Candidate continuation text.
        token_count: Number of generated tokens for the candidate.
        closed_with_tag: Whether candidate contains `</steer>`.
        finish_reason: Finish reason returned by vLLM.
        cumulative_logprob: Sum of selected token logprobs.
        average_logprob: Mean selected token logprob.

    Returns:
        Dataclass storing one candidate summary.
    """

    step_index: int
    candidate_index: int
    text: str
    token_count: int
    closed_with_tag: bool
    finish_reason: str
    cumulative_logprob: float
    average_logprob: float

    def preview(self, *, limit: int = 80) -> str:
        """Create a compact preview for logs and UI tables.

        Args:
            limit: Maximum output characters.

        Returns:
            Truncated one-line candidate preview.
        """
        compact_text = self.text.replace("\n", " ").strip()
        if len(compact_text) <= limit:
            return compact_text
        return compact_text[:limit] + "..."


@dataclass(frozen=True)
class BranchStep:
    """Branch event metadata for one rollout step.

    Args:
        step_index: Zero-based branch step index.
        prefix_char_end: Character offset where branch started.
        selected_candidate_index: Candidate selected for continuation.
        selected_text: Selected continuation text.
        total_candidates: Number of candidates sampled at branch point.
        unique_candidate_count: Number of unique candidate texts.
        terminated: Whether run ended at this step.
        termination_reason: Terminal reason for this step.

    Returns:
        Dataclass storing one branching step summary.
    """

    step_index: int
    prefix_char_end: int
    selected_candidate_index: int
    selected_text: str
    total_candidates: int
    unique_candidate_count: int
    terminated: bool
    termination_reason: str

    def summary(self) -> str:
        """Create canonical summary text.

        Args:
            None.

        Returns:
            Human-readable one-line summary.
        """
        return (
            f"step={self.step_index}, selected={self.selected_candidate_index}, "
            f"unique={self.unique_candidate_count}/{self.total_candidates}, "
            f"terminated={self.terminated}, reason={self.termination_reason}"
        )


@dataclass(frozen=True)
class RunArtifactsIndex:
    """Canonical output file index for one run.

    Args:
        run_id: Stable run identifier string.
        run_dir: Root output directory for run artifacts.
        config_path: Run config JSON path.
        steps_path: Step metadata JSONL path.
        candidates_path: Candidate metadata JSONL path.
        token_stats_path: Token statistics JSONL path.
        report_path: Static HTML report path.

    Returns:
        Dataclass storing canonical artifact paths.
    """

    run_id: str
    run_dir: Path
    config_path: Path
    steps_path: Path
    candidates_path: Path
    token_stats_path: Path
    report_path: Path

    def as_dict(self) -> dict[str, str]:
        """Serialize artifact paths to JSON-friendly mapping.

        Args:
            None.

        Returns:
            Mapping from artifact name to string path.
        """
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "config_path": str(self.config_path),
            "steps_path": str(self.steps_path),
            "candidates_path": str(self.candidates_path),
            "token_stats_path": str(self.token_stats_path),
            "report_path": str(self.report_path),
        }
