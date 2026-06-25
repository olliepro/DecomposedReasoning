#!/usr/bin/env python3
"""Probe verbalized off-policy sampling against a completions endpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://a0111:18001"
DEFAULT_MODEL = "qwen35-4b-5611097-step300"
DEFAULT_PREFIX = "<think>\n"


def request_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str],
) -> str:
    """Return one non-streaming completion text."""

    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
    }
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        print(error.read().decode("utf-8", errors="replace"), file=sys.stderr)
        raise
    return str(payload["choices"][0]["text"])


def main() -> None:
    """Run one enumerate-plus-continue debug probe."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--min-candidates", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--fanout", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enumeration-tokens", type=int, default=512)
    parser.add_argument("--continuation-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    candidate_count = rng.randint(args.min_candidates, args.max_candidates)
    assert args.fanout <= candidate_count

    enumerate_steer = (
        f"\n<steer>Enumerate {candidate_count} distinct options for the "
        "immediate next decision/step</steer>\n<exec>"
    )
    enumeration_prompt = f"{args.prefix}{enumerate_steer}"
    enumeration = request_completion(
        base_url=args.base_url,
        model=args.model,
        prompt=enumeration_prompt,
        max_tokens=args.enumeration_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=["</exec"],
    )
    print("=== enumeration ===")
    print(enumeration)

    option_numbers = rng.sample(range(1, candidate_count + 1), k=args.fanout)
    for option_number in option_numbers:
        continue_steer = (
            f"</exec>\n<steer>Proceed with option {option_number}</steer>\n"
            f"<exec>Let's do option {option_number}:"
        )
        continuation = request_completion(
            base_url=args.base_url,
            model=args.model,
            prompt=f"{enumeration_prompt}{enumeration}{continue_steer}",
            max_tokens=args.continuation_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=["</exec"],
        )
        print(f"\n=== continue #{option_number} ===")
        print(continuation)


if __name__ == "__main__":
    main()
