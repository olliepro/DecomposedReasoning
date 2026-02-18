"""CLI entrypoint for steer-aware branching analysis with vLLM."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TypedDict

from analysis_types import RunConfig
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
        max_server_logprobs: Server-side top-logprob cap.
        seed: RNG seed.
        rollout_chunk_tokens: Rollout chunk size.

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
    max_server_logprobs: int
    seed: int
    rollout_chunk_tokens: int


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
    parser.add_argument("--rollout-chunk-tokens", type=int, default=512)
    parser.add_argument(
        "--log-level",
        type=str,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )


def configure_logging(*, log_level: str) -> None:
    """Configure process logging for stage-level rollout tracing.

    Args:
        log_level: Logging threshold name.

    Returns:
        None.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


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


def build_run_config(*, args: argparse.Namespace) -> RunConfig:
    """Build validated `RunConfig` from CLI namespace.

    Args:
        args: Parsed CLI args.

    Returns:
        Runtime configuration object.
    """
    prompt = resolve_prompt(prompt=args.prompt, prompt_file=args.prompt_file)
    run_kwargs = build_run_kwargs(args=args, prompt=prompt)
    config = RunConfig(**run_kwargs)
    config.validate()
    return config


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
        "max_server_logprobs": int(args.max_server_logprobs),
        "seed": int(args.seed),
        "rollout_chunk_tokens": int(args.rollout_chunk_tokens),
    }


def main() -> None:
    """Run CLI pipeline and build report viewer.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    configure_logging(log_level=str(args.log_level))
    config = build_run_config(args=args)
    artifacts = run_branching_analysis(config=config)
    report_path = build_report(
        run_dirs=[artifacts.run_dir], output_path=artifacts.report_path
    )
    print(str(artifacts.run_dir))
    print(str(report_path))


if __name__ == "__main__":
    main()
