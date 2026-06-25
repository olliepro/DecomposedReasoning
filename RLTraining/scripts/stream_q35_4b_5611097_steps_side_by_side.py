#!/usr/bin/env python3
"""Stream the two added-token SFT vLLM responses side-by-side."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_LEFT_JOB_ID = "6121191"
DEFAULT_RIGHT_JOB_ID = "6121192"
DEFAULT_LEFT_MODEL = "qwen35-4b-instruct-addedtok-sft-8ep"
DEFAULT_RIGHT_MODEL = "qwen35-4b-base-addedtok-sft-8ep"
DEFAULT_LEFT_PORT = 18020
DEFAULT_RIGHT_PORT = 18021
ANSI_RESET = "\033[0m"
STEER_TOKEN_COLOR = "\033[1;96m"
EXEC_TOKEN_COLOR = "\033[1;93m"
CONTROL_TOKEN_COLORS = {
    "<steer>": STEER_TOKEN_COLOR,
    "</steer>": STEER_TOKEN_COLOR,
    "<exec>": EXEC_TOKEN_COLOR,
    "</exec>": EXEC_TOKEN_COLOR,
}
DEFAULT_SYSTEM_PROMPT = (
    "Solve the math problem. Put your reasoning in one <think>...</think> "
    "block made of alternating non-empty <steer>...</steer> and "
    "<exec>...</exec> blocks, starting with <steer>. Use <steer> blocks for "
    "guidance: guide thinking, make executive decisions, choose subproblems, "
    "slow down, enumerate, verify, or backtrack. Examples: Guide thinking: "
    '"Try applying ___." Make decisions: "Name the dog \'___\'." Choose '
    'subproblems: "Consider a<=3." Slow down: "Use a more precise method." '
    'Enumerate: "List 5 options and choose one." Verify: '
    '"Double Check that calculation." Backtrack: "Abandon this approach." '
    "Use <exec> blocks to precisely carry out the chosen guidance with "
    "calculations and deductions. After </think>, give the final answer as "
    "\\boxed{...} with no extra prose."
)


@dataclass(frozen=True)
class StreamTarget:
    label: str
    model: str
    base_url: str


@dataclass(frozen=True)
class StreamEvent:
    side: str
    text: str = ""
    done: bool = False
    error: str = ""


@dataclass
class ScrollBuffers:
    left: str = ""
    right: str = ""


@dataclass
class ScrollLines:
    left: list[str]
    right: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt", nargs="*", help="Prompt text. Reads stdin if omitted."
    )
    parser.add_argument("--left-job-id", default=DEFAULT_LEFT_JOB_ID)
    parser.add_argument("--right-job-id", default=DEFAULT_RIGHT_JOB_ID)
    parser.add_argument("--left-base-url", default="")
    parser.add_argument("--right-base-url", default="")
    parser.add_argument("--left-model", default=DEFAULT_LEFT_MODEL)
    parser.add_argument("--right-model", default=DEFAULT_RIGHT_MODEL)
    parser.add_argument("--left-port", type=int, default=DEFAULT_LEFT_PORT)
    parser.add_argument("--right-port", type=int, default=DEFAULT_RIGHT_PORT)
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt rendered before the user prompt in both requests.",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Render the chat-template prompt without a system message.",
    )
    parser.add_argument(
        "--assistant-prefix",
        default="<think>\n",
        help="Partial assistant completion appended before streaming.",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Send the prompt exactly as provided to /v1/completions.",
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--refresh-seconds", type=float, default=0.08)
    parser.add_argument("--scroll-flush-seconds", type=float, default=1.0)
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Redraw a fixed terminal dashboard instead of preserving scrollback.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI highlighting for added control tokens.",
    )
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


def base_url_for_job(*, job_id: str, port: int) -> str:
    return f"http://{node_for_job(job_id=job_id)}:{port}/v1"


def build_targets(*, args: argparse.Namespace) -> tuple[StreamTarget, StreamTarget]:
    left_url = args.left_base_url or base_url_for_job(
        job_id=args.left_job_id,
        port=args.left_port,
    )
    right_url = args.right_base_url or base_url_for_job(
        job_id=args.right_job_id,
        port=args.right_port,
    )
    return (
        StreamTarget(
            label="instruct added-token SFT", model=args.left_model, base_url=left_url
        ),
        StreamTarget(
            label="base added-token SFT", model=args.right_model, base_url=right_url
        ),
    )


def render_prompt(*, prompt: str, args: argparse.Namespace) -> str:
    if args.raw_prompt:
        return prompt
    system_part = ""
    if not args.no_system_prompt:
        system_part = f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n"
    return (
        system_part
        + f"<|im_start|>user\n{prompt}<|im_end|>\n"
        + f"<|im_start|>assistant\n{args.assistant_prefix}"
    )


def request_body(*, model: str, prompt: str, args: argparse.Namespace) -> bytes:
    payload = {
        "model": model,
        "prompt": render_prompt(prompt=prompt, args=args),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": True,
    }
    return json.dumps(payload).encode("utf-8")


def stream_target(
    *,
    side: str,
    target: StreamTarget,
    prompt: str,
    args: argparse.Namespace,
    events: queue.Queue[StreamEvent],
) -> None:
    request = urllib.request.Request(
        f"{target.base_url.rstrip('/')}/completions",
        data=request_body(model=target.model, prompt=prompt, args=args),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10800) as response:
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
                    events.put(StreamEvent(side=side, text=text))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        events.put(StreamEvent(side=side, error=f"HTTP {exc.code}: {detail}"))
    except Exception as exc:
        events.put(StreamEvent(side=side, error=f"{type(exc).__name__}: {exc}"))
    finally:
        events.put(StreamEvent(side=side, done=True))


def wrap_cell(*, text: str, width: int, height: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines() or [""]:
        lines.extend(
            textwrap.wrap(raw_line, width=width, replace_whitespace=False) or [""]
        )
    return lines[-height:]


def color_enabled(*, args: argparse.Namespace) -> bool:
    return not args.no_color and "NO_COLOR" not in os.environ


def highlight_control_tokens(*, text: str, enabled: bool) -> str:
    if not enabled:
        return text
    highlighted = text
    for token, color in CONTROL_TOKEN_COLORS.items():
        highlighted = highlighted.replace(token, f"{color}{token}{ANSI_RESET}")
    return highlighted


def format_cell(*, text: str, width: int, color: bool) -> str:
    padded = f"{text:<{width}}"
    return highlight_control_tokens(text=padded, enabled=color)


def render_columns(
    *,
    left_target: StreamTarget,
    right_target: StreamTarget,
    left_text: str,
    right_text: str,
    left_done: bool,
    right_done: bool,
    temperature: float,
    color: bool,
) -> None:
    term = shutil.get_terminal_size(fallback=(160, 48))
    width = max(30, (term.columns - 5) // 2)
    height = max(12, term.lines - 6)
    left_lines = wrap_cell(text=left_text, width=width, height=height)
    right_lines = wrap_cell(text=right_text, width=width, height=height)
    line_count = max(len(left_lines), len(right_lines), 1)
    left_header = f"{left_target.label} [{left_target.model}]"
    right_header = f"{right_target.label} [{right_target.model}]"
    status = f"{'done' if left_done else 'streaming'} | {'done' if right_done else 'streaming'}"
    print("\033[H\033[J", end="")
    print(f"{left_header[:width]:<{width}}  || {right_header[:width]:<{width}}")
    print(f"{status[:width]:<{width}}  || {f'temperature={temperature:g}':<{width}}")
    print(f"{'-' * width}  || {'-' * width}")
    for index in range(line_count):
        left = left_lines[index] if index < len(left_lines) else ""
        right = right_lines[index] if index < len(right_lines) else ""
        print(
            f"{format_cell(text=left, width=width, color=color)}  || "
            f"{format_cell(text=right, width=width, color=color)}"
        )
    sys.stdout.flush()


def column_width() -> int:
    term = shutil.get_terminal_size(fallback=(160, 48))
    return max(30, (term.columns - 5) // 2)


def split_width_line(*, text: str, width: int) -> tuple[str, str]:
    if len(text) <= width:
        return text, ""
    break_at = text.rfind(" ", 0, width + 1)
    if break_at < max(12, width // 3):
        break_at = width
    return text[:break_at], text[break_at:].lstrip(" ")


def pop_ready_lines(
    *, text: str, width: int, force: bool = False
) -> tuple[list[str], str]:
    lines: list[str] = []
    remainder = text
    while "\n" in remainder:
        raw_line, remainder = remainder.split("\n", maxsplit=1)
        lines.extend(wrap_stream_line(text=raw_line, width=width))
    while len(remainder) >= width:
        line, remainder = split_width_line(text=remainder, width=width)
        lines.append(line)
    if force and remainder:
        lines.append(remainder)
        remainder = ""
    return lines, remainder


def wrap_stream_line(*, text: str, width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    remainder = text
    while remainder:
        line, remainder = split_width_line(text=remainder, width=width)
        lines.append(line)
    return lines


def print_scroll_header(
    *,
    left_target: StreamTarget,
    right_target: StreamTarget,
    temperature: float,
) -> None:
    width = column_width()
    left_header = f"{left_target.label} [{left_target.model}]"
    right_header = f"{right_target.label} [{right_target.model}]"
    print(f"{left_header[:width]:<{width}}  || {right_header[:width]:<{width}}")
    temp_label = f"temperature={temperature:g}"
    print(f"{temp_label:<{width}}  || {temp_label:<{width}}")
    print(f"{'-' * width}  || {'-' * width}", flush=True)


def append_scroll_text(
    *,
    buffers: ScrollBuffers,
    ready: ScrollLines,
    side: str,
    text: str = "",
    force: bool = False,
) -> None:
    width = column_width()
    if side == "left":
        buffers.left += text
        lines, buffers.left = pop_ready_lines(
            text=buffers.left, width=width, force=force
        )
        ready.left.extend(lines)
    else:
        buffers.right += text
        lines, buffers.right = pop_ready_lines(
            text=buffers.right,
            width=width,
            force=force,
        )
        ready.right.extend(lines)


def flush_paired_rows(*, ready: ScrollLines, color: bool, force: bool = False) -> None:
    width = column_width()
    printed = False
    while ready.left and ready.right:
        left = ready.left.pop(0)
        right = ready.right.pop(0)
        print(
            f"{format_cell(text=left, width=width, color=color)}  || "
            f"{format_cell(text=right, width=width, color=color)}"
        )
        printed = True
    if force:
        while ready.left or ready.right:
            left = ready.left.pop(0) if ready.left else ""
            right = ready.right.pop(0) if ready.right else ""
            print(
                f"{format_cell(text=left, width=width, color=color)}  || "
                f"{format_cell(text=right, width=width, color=color)}"
            )
            printed = True
    if printed:
        sys.stdout.flush()


def flush_unpaired_backlog(
    *, ready: ScrollLines, color: bool, max_backlog: int = 12
) -> None:
    width = column_width()
    printed = False
    while len(ready.left) > max_backlog:
        left = ready.left.pop(0)
        print(f"{format_cell(text=left, width=width, color=color)}  || {'':<{width}}")
        printed = True
    while len(ready.right) > max_backlog:
        right = ready.right.pop(0)
        print(f"{'':<{width}}  || {format_cell(text=right, width=width, color=color)}")
        printed = True
    if printed:
        sys.stdout.flush()


def print_scroll_done(*, left_done: bool, right_done: bool) -> None:
    width = column_width()
    left_status = "done" if left_done else "streaming"
    right_status = "done" if right_done else "streaming"
    print(f"{left_status:<{width}}  || {right_status:<{width}}", flush=True)


def run_streams(
    *,
    left_target: StreamTarget,
    right_target: StreamTarget,
    prompt: str,
    args: argparse.Namespace,
) -> None:
    events: queue.Queue[StreamEvent] = queue.Queue()
    threads = [
        threading.Thread(
            target=stream_target,
            kwargs={
                "side": "left",
                "target": left_target,
                "prompt": prompt,
                "args": args,
                "events": events,
            },
            daemon=True,
        ),
        threading.Thread(
            target=stream_target,
            kwargs={
                "side": "right",
                "target": right_target,
                "prompt": prompt,
                "args": args,
                "events": events,
            },
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    left_text = ""
    right_text = ""
    left_done = False
    right_done = False
    use_color = color_enabled(args=args)
    last_render = 0.0
    last_scroll_flush = time.monotonic()
    scroll_buffers = ScrollBuffers()
    scroll_lines = ScrollLines(left=[], right=[])
    if not args.dashboard:
        print_scroll_header(
            left_target=left_target,
            right_target=right_target,
            temperature=args.temperature,
        )
    while not (left_done and right_done):
        try:
            event = events.get(timeout=args.refresh_seconds)
        except queue.Empty:
            event = None
        if event is not None:
            delta = event.error or event.text
            if event.side == "left":
                left_text += delta
                left_done = left_done or event.done
            else:
                right_text += delta
                right_done = right_done or event.done
            if not args.dashboard:
                append_scroll_text(
                    buffers=scroll_buffers,
                    ready=scroll_lines,
                    side=event.side,
                    text=delta,
                )
                if scroll_lines.left and scroll_lines.right:
                    flush_paired_rows(ready=scroll_lines, color=use_color)
                    last_scroll_flush = time.monotonic()
        now = time.monotonic()
        if args.dashboard and now - last_render >= args.refresh_seconds:
            render_columns(
                left_target=left_target,
                right_target=right_target,
                left_text=left_text,
                right_text=right_text,
                left_done=left_done,
                right_done=right_done,
                temperature=args.temperature,
                color=use_color,
            )
            last_render = now
        if (
            not args.dashboard
            and (scroll_lines.left or scroll_lines.right)
            and now - last_scroll_flush >= args.scroll_flush_seconds
        ):
            flush_paired_rows(ready=scroll_lines, color=use_color)
            flush_unpaired_backlog(ready=scroll_lines, color=use_color)
            last_scroll_flush = now
    if args.dashboard:
        render_columns(
            left_target=left_target,
            right_target=right_target,
            left_text=left_text,
            right_text=right_text,
            left_done=left_done,
            right_done=right_done,
            temperature=args.temperature,
            color=use_color,
        )
    else:
        append_scroll_text(
            buffers=scroll_buffers,
            ready=scroll_lines,
            side="left",
            force=True,
        )
        append_scroll_text(
            buffers=scroll_buffers,
            ready=scroll_lines,
            side="right",
            force=True,
        )
        flush_paired_rows(ready=scroll_lines, color=use_color, force=True)
        print_scroll_done(left_done=left_done, right_done=right_done)


def main() -> None:
    args = parse_args()
    prompt = " ".join(args.prompt).strip() if args.prompt else sys.stdin.read().strip()
    assert prompt, "provide a prompt as args or stdin"
    left_target, right_target = build_targets(args=args)
    run_streams(
        left_target=left_target,
        right_target=right_target,
        prompt=prompt,
        args=args,
    )


if __name__ == "__main__":
    main()
