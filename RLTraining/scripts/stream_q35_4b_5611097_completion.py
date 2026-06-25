#!/usr/bin/env python3
"""Stream a /v1/completions response from the step-240 vLLM server."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_JOB_ID = "5854283"
DEFAULT_MODEL = "qwen35-4b-5611097-step240"
DEFAULT_PORT = 18000
DEFAULT_SYSTEM_PROMPT = (
    "Solve the task. Put your reasoning in one <think>...</think> "
    "block made of alternating non-empty <steer>...</steer> and "
    "<exec>...</exec> blocks, starting with <steer>. Use <steer> blocks "
    "to guide thinking, make executive decisions, choose subproblems, slow "
    "down, enumerate, verify, or backtrack. Examples: Guide thinking: "
    '"Try applying ___." Make decisions: "Name the dog \'___\'." '
    'Choose subproblems: "Consider a<=3." Slow down: '
    '"Use a more precise method." Enumerate: "List 5 options and choose '
    'one." Verify: "Double Check that calculation." Backtrack: '
    '"Abandon this approach." Use <exec> blocks to precisely carry out the '
    "chosen guidance with concrete work and deductions. After </think>, give "
    "the final answer clearly and concisely."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="*",
        default=["Hello"],
        help="User prompt text. Defaults to 'Hello'. Reads stdin with --stdin.",
    )
    parser.add_argument("--stdin", action="store_true", help="Read prompt from stdin.")
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Send the prompt exactly as provided instead of applying the chat template.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt rendered before the user message.",
    )
    parser.add_argument(
        "--partial-completion",
        "--assistant-prefix",
        dest="assistant_prefix",
        default="",
        help=(
            "Optional partial model completion appended after the rendered user "
            "message, before streaming the continuation."
        ),
    )
    parser.add_argument(
        "--print-rendered-prompt",
        action="store_true",
        help="Print the exact prompt sent to /v1/completions and exit.",
    )
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--stop", action="append", default=[])
    return parser.parse_args()


def render_prompt(*, args: argparse.Namespace, prompt: str) -> str:
    if args.raw_prompt:
        return prompt

    rendered = (
        f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    return f"{rendered}{args.assistant_prefix}"


def node_for_job(*, job_id: str) -> str:
    output = subprocess.check_output(
        ["squeue", "-j", job_id, "-h", "-o", "%T|%R"],
        text=True,
    ).strip()
    assert output, f"job {job_id} is not in squeue"
    state, node = output.split("|", maxsplit=1)
    assert state == "RUNNING", f"job {job_id} is {state}: {node}"
    return node.split(",", maxsplit=1)[0]


def request_body(*, args: argparse.Namespace, prompt: str) -> bytes:
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": True,
    }
    if args.stop:
        payload["stop"] = args.stop
    return json.dumps(payload).encode("utf-8")


def stream_completion(*, base_url: str, body: bytes) -> None:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/completions",
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
            text = chunk["choices"][0].get("text", "")
            if text:
                print(text, end="", flush=True)
    print()


def main() -> None:
    args = parse_args()
    prompt = sys.stdin.read() if args.stdin else " ".join(args.prompt)
    assert prompt, "provide a prompt as args or stdin"
    rendered_prompt = render_prompt(args=args, prompt=prompt)
    if args.print_rendered_prompt:
        print(rendered_prompt, end="" if rendered_prompt.endswith("\n") else "\n")
        return

    base_url = args.base_url
    if not base_url:
        base_url = f"http://{node_for_job(job_id=args.job_id)}:{args.port}/v1"
    stream_completion(
        base_url=base_url,
        body=request_body(args=args, prompt=rendered_prompt),
    )


if __name__ == "__main__":
    main()
