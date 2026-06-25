#!/usr/bin/env python3
"""Stream a response from the qwen35 4B 5553508 step-200 vLLM server."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_JOB_ID = "5689251"
DEFAULT_MODEL = "qwen35-4b-5553508-step200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt", nargs="*", help="Prompt text. Reads stdin if omitted."
    )
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def node_for_job(*, job_id: str) -> str:
    output = subprocess.check_output(
        ["squeue", "-j", job_id, "-h", "-o", "%T|%R"],
        text=True,
    ).strip()
    assert output, f"job {job_id} is not in squeue"
    state, node = output.split("|", maxsplit=1)
    assert state == "RUNNING", f"job {job_id} is {state}: {node}"
    return node.split(",", maxsplit=1)[0]


def request_payload(*, args: argparse.Namespace, prompt: str) -> bytes:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": True,
    }
    return json.dumps(payload).encode("utf-8")


def stream_response(*, base_url: str, body: bytes) -> None:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=10800)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} from vLLM:\n{detail}") from exc
    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content")
            if text:
                print(text, end="", flush=True)
    print()


def main() -> None:
    args = parse_args()
    prompt = " ".join(args.prompt).strip() if args.prompt else sys.stdin.read().strip()
    assert prompt, "provide a prompt as args or stdin"
    base_url = args.base_url
    if not base_url:
        base_url = f"http://{node_for_job(job_id=args.job_id)}:18000/v1"
    stream_response(base_url=base_url, body=request_payload(args=args, prompt=prompt))


if __name__ == "__main__":
    main()
