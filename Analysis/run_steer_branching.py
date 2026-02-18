"""CLI entrypoint for steer-aware branching analysis with vLLM."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict

from analysis_types import ApiModeConfig, RunConfig, TemplateConfig
from branching_engine import run_branching_analysis
from build_report import build_report


class RunKwargs(TypedDict):
    """Typed kwargs used to instantiate `RunConfig`.

    Args:
        base_url: Base OpenAI-compatible URL.
        model: Model identifier.
        prompt: User prompt.
        output_root: Output root directory.
        temperature: Sampling temperature.
        top_p: Nucleus sampling.
        max_total_tokens: Run token budget.
        branch_factor: Number of steer candidates.
        n_keep: Number of kept branches.
        max_steer_tokens: Max steer tokens.
        max_steps: Max branch steps.
        top_logprobs: Requested top alternatives.
        seed: RNG seed.
        rollout_chunk_tokens: Rollout chunk size.
        boundary_pattern: Branch regex pattern.

    Returns:
        Typed dictionary for safe `RunConfig` construction.
    """

    base_url: str
    model: str
    prompt: str
    output_root: Path
    temperature: float
    top_p: float
    max_total_tokens: int
    branch_factor: int
    n_keep: int
    max_steer_tokens: int
    max_steps: int
    top_logprobs: int
    seed: int
    rollout_chunk_tokens: int
    boundary_pattern: str


def parse_args() -> argparse.Namespace:
    """Parse CLI args for steer-branching run.

    Args:
        None.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Run steer-aware branching analysis.")
    add_core_args(parser=parser)
    add_sampling_args(parser=parser)
    add_template_args(parser=parser)
    return parser.parse_args()


def add_core_args(*, parser: argparse.ArgumentParser) -> None:
    """Add core API and prompt arguments to parser.

    Args:
        parser: CLI argument parser.

    Returns:
        None.
    """
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument(
        "--api-mode", choices=["completions", "chat"], default="completions"
    )
    parser.add_argument(
        "--allow-fallback", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-server-logprobs", type=int, default=20)


def add_sampling_args(*, parser: argparse.ArgumentParser) -> None:
    """Add sampling and branching arguments to parser.

    Args:
        parser: CLI argument parser.

    Returns:
        None.
    """
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    parser.add_argument("--max-total-tokens", type=int, default=32768)
    parser.add_argument("--branch-factor", type=int, default=100)
    parser.add_argument("--n-keep", type=int, default=1)
    parser.add_argument("--max-steer-tokens", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-chunk-tokens", type=int, default=256)
    parser.add_argument("--boundary-pattern", type=str, default=r"<steer\b[^>]*>")


def add_template_args(*, parser: argparse.ArgumentParser) -> None:
    """Add chat template options to parser.

    Args:
        parser: CLI argument parser.

    Returns:
        None.
    """
    parser.add_argument(
        "--add-generation-prompt", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--continue-final-message", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--chat-template", type=str, default=None)
    parser.add_argument("--chat-template-kwarg", action="append", default=[])
    parser.add_argument(
        "--content-format", choices=["string", "openai"], default="string"
    )


def resolve_prompt(*, prompt: str | None, prompt_file: Path | None) -> str:
    """Resolve prompt value from direct input or file.

    Args:
        prompt: Prompt text from CLI.
        prompt_file: Optional prompt file path.

    Returns:
        Prompt text.
    """
    if prompt is not None and prompt_file is not None:
        raise SystemExit("Provide exactly one of --prompt or --prompt-file")
    if prompt is not None:
        return prompt
    if prompt_file is None:
        raise SystemExit("Provide one of --prompt or --prompt-file")
    return prompt_file.read_text(encoding="utf-8")


def parse_template_kwargs(*, items: list[str]) -> dict[str, object]:
    """Parse repeated `key=value` pairs into mapping.

    Args:
        items: CLI key/value string entries.

    Returns:
        Parsed key/value mapping.
    """
    parsed: dict[str, object] = {}
    for item in items:
        assert "=" in item, "template kwargs must use key=value"
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def build_run_config(*, args: argparse.Namespace) -> RunConfig:
    """Build validated `RunConfig` from CLI namespace.

    Args:
        args: Parsed CLI args.

    Returns:
        Runtime configuration object.
    """
    prompt = resolve_prompt(prompt=args.prompt, prompt_file=args.prompt_file)
    api_mode_config = build_api_mode_config(args=args)
    template_config = build_template_config(args=args)
    run_kwargs = build_run_kwargs(args=args, prompt=prompt)
    config = RunConfig(
        api_mode_config=api_mode_config,
        template_config=template_config,
        **run_kwargs,
    )
    config.validate()
    return config


def build_api_mode_config(*, args: argparse.Namespace) -> ApiModeConfig:
    """Build API mode configuration from CLI args.

    Args:
        args: Parsed CLI args.

    Returns:
        API mode configuration.
    """
    return ApiModeConfig(
        default_mode=args.api_mode,
        allow_fallback=bool(args.allow_fallback),
        max_server_logprobs=int(args.max_server_logprobs),
    )


def build_template_config(*, args: argparse.Namespace) -> TemplateConfig:
    """Build template configuration from CLI args.

    Args:
        args: Parsed CLI args.

    Returns:
        Template configuration object.
    """
    return TemplateConfig(
        use_raw_im_template=True,
        add_generation_prompt=bool(args.add_generation_prompt),
        continue_final_message=bool(args.continue_final_message),
        chat_template=args.chat_template,
        chat_template_kwargs=parse_template_kwargs(
            items=list(args.chat_template_kwarg)
        ),
        content_format=args.content_format,
    )


def build_run_kwargs(*, args: argparse.Namespace, prompt: str) -> RunKwargs:
    """Build plain run kwargs used to instantiate `RunConfig`.

    Args:
        args: Parsed CLI args.
        prompt: Resolved prompt text.

    Returns:
        Mapping of `RunConfig` constructor keyword arguments.
    """
    return {
        "base_url": args.base_url,
        "model": args.model,
        "prompt": prompt,
        "output_root": args.output_root,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_total_tokens": int(args.max_total_tokens),
        "branch_factor": int(args.branch_factor),
        "n_keep": int(args.n_keep),
        "max_steer_tokens": int(args.max_steer_tokens),
        "max_steps": int(args.max_steps),
        "top_logprobs": int(args.top_logprobs),
        "seed": int(args.seed),
        "rollout_chunk_tokens": int(args.rollout_chunk_tokens),
        "boundary_pattern": args.boundary_pattern,
    }


def main() -> None:
    """Run CLI pipeline and build report viewer.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    config = build_run_config(args=args)
    artifacts = run_branching_analysis(config=config)
    report_path = build_report(
        run_dirs=[artifacts.run_dir], output_path=artifacts.report_path
    )
    print(str(artifacts.run_dir))
    print(str(report_path))


if __name__ == "__main__":
    main()
