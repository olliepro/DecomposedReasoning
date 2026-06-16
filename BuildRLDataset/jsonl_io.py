from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def decode_json_line(line: str, line_number: int) -> list[dict[str, object]]:
    """Decode one JSONL line into one or more JSON objects."""

    decoder = json.JSONDecoder()
    index = 0
    objects: list[dict[str, object]] = []
    while index < len(line):
        while index < len(line) and line[index].isspace():
            index += 1
        if index >= len(line):
            break
        obj, end = decoder.raw_decode(line, index)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected object at line {line_number} offset {index}")
        objects.append(obj)
        index = end
    return objects


def write_jsonl_row(output_path: Path, row: dict[str, object]) -> None:
    """Append one JSON object to a JSONL file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def count_jsonl_rows(path: Path) -> int:
    """Count non-empty rows in a JSONL file."""

    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Yield JSON objects from a JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            for row in decode_json_line(line=line, line_number=line_number):
                yield row
