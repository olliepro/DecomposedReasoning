from __future__ import annotations

import json
import re
import xml.etree.ElementTree as element_tree
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
TAG_BLOCK_PATTERN = re.compile(
    r"<(steer|steering|exec|execute|execution)>(.*?)</\1>",
    flags=re.DOTALL | re.IGNORECASE,
)


def cache_data_if_available(func):
    """Apply `st.cache_data` only within an active Streamlit context.

    Args:
        func: Function to potentially wrap with Streamlit cache.

    Returns:
        Cached function in-app, plain function otherwise.
    """
    try:
        runtime_module = getattr(st, "runtime", None)
        exists_fn = getattr(runtime_module, "exists", None)
        if runtime_module is None or not callable(exists_fn) or not bool(exists_fn()):
            return func
    except Exception:  # noqa: BLE001
        return func
    return st.cache_data(show_spinner=False)(func)


@dataclass(frozen=True)
class SteeringSection:
    """One steering/execution pair.

    Args:
        steer_text: Steering directive text.
        execute_text: Execution body text.
    """

    steer_text: str
    execute_text: str

    def label(self, index: int) -> str:
        """Create a compact heading label.

        Args:
            index: 1-based section index.

        Returns:
            Formatted section title.
        """
        return f"{index}. {self.steer_text or '(empty steering)'}"


@dataclass(frozen=True)
class ThinkBlockRef:
    """Reference to a think block within one row.

    Args:
        message_index: Message index in `messages`.
        block_index: Think block index within assistant message.
        block_text: Raw think content.
    """

    message_index: int
    block_index: int
    block_text: str

    def label(self) -> str:
        """Build a stable selector label.

        Returns:
            Sidebar label for this think block.
        """
        preview = self.block_text.strip().replace("\n", " ")
        preview = preview[:60] + ("..." if len(preview) > 60 else "")
        return f"message {self.message_index}, think {self.block_index}: {preview}"


@dataclass(frozen=True)
class RowOption:
    """Sidebar row selector option.

    Args:
        row_index: 0-based row index.
        row_id: Dataset row identifier.
        dataset_source: Source label.
        think_count: Number of think blocks in row.
    """

    row_index: int
    row_id: str
    dataset_source: str
    think_count: int

    def label(self) -> str:
        """Build row selector label.

        Returns:
            Human-readable row label.
        """
        return (
            f"row {self.row_index} | id={self.row_id} | "
            f"source={self.dataset_source} | think={self.think_count}"
        )


def discover_dataset_files(base_dir: Path) -> list[Path]:
    """Discover transformed JSONL files.

    Args:
        base_dir: Project directory containing output folders.

    Returns:
        Existing candidate dataset files sorted by priority.

    Example:
        >>> discover_dataset_files(base_dir=Path('.'))  # doctest: +SKIP
    """
    preferred = [
        base_dir / "output_nonbatch5" / "transformed_output.jsonl",
        base_dir / "output" / "transformed_output.jsonl",
    ]
    discovered: list[Path] = []
    for path in preferred:
        if path.exists():
            discovered.append(path)

    globbed = sorted(base_dir.glob("output*/**/*transformed*.jsonl"))
    for path in globbed:
        if path not in discovered:
            discovered.append(path)
    return discovered


@cache_data_if_available
def load_jsonl_rows(path_str: str) -> list[dict[str, object]]:
    """Load one JSON object per line from JSONL.

    Args:
        path_str: JSONL path as string.

    Returns:
        Parsed rows.
    """
    path = Path(path_str)
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def extract_think_refs(messages: list[dict[str, object]]) -> list[ThinkBlockRef]:
    """Extract think-block references from assistant messages.

    Args:
        messages: Row messages list.

    Returns:
        Think block references.
    """
    refs: list[ThinkBlockRef] = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        blocks = THINK_BLOCK_PATTERN.findall(content)
        for block_index, block_text in enumerate(blocks):
            refs.append(
                ThinkBlockRef(
                    message_index=message_index,
                    block_index=block_index,
                    block_text=block_text,
                )
            )
    return refs


def build_row_options(rows: list[dict[str, object]]) -> list[RowOption]:
    """Build row options that contain at least one think block.

    Args:
        rows: Parsed dataset rows.

    Returns:
        Selectable row options.
    """
    options: list[RowOption] = []
    for row_index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        think_count = len(extract_think_refs(messages=messages))
        if think_count == 0:
            continue
        options.append(
            RowOption(
                row_index=row_index,
                row_id=str(row.get("id", f"row-{row_index}")),
                dataset_source=str(row.get("dataset_source", "unknown")),
                think_count=think_count,
            )
        )
    return options


def strip_code_fences(text: str) -> str:
    """Strip optional markdown fences around XML text.

    Args:
        text: Raw think text.

    Returns:
        Fence-free text.
    """
    cleaned = text.lstrip("\ufeff").strip()
    if "```" in cleaned:
        first_fence = cleaned.find("```")
        cleaned = cleaned[first_fence:]
        cleaned = re.sub(r"^```(?:[a-zA-Z0-9_-]+)?\s*", "", cleaned, count=1)
        if "```" in cleaned:
            cleaned = cleaned[: cleaned.rfind("```")]
    return cleaned.strip()


def parse_xml_sections(xml_text: str) -> tuple[list[SteeringSection], str | None]:
    """Parse steering/exec sections from XML-like text.

    Args:
        xml_text: Think text containing steer/exec-style tags.

    Returns:
        Tuple of parsed sections and optional parse warning.

    Example:
        >>> parse_xml_sections('<steer>A</steer><exec>B</exec>')[0][0].steer_text
        'A'
    """
    cleaned = strip_code_fences(text=xml_text)
    wrapped = f"<root>{cleaned}</root>"
    sections: list[SteeringSection] = []

    try:
        root = element_tree.fromstring(wrapped)
        for child in list(root):
            tag = child.tag.lower().split("}")[-1]
            text = "".join(child.itertext()).strip()
            if tag in {"steer", "steering"}:
                sections.append(SteeringSection(steer_text=text, execute_text=""))
                continue
            if tag in {"exec", "execute", "execution"}:
                if not sections:
                    sections.append(
                        SteeringSection(
                            steer_text="Unlabeled execution",
                            execute_text=text,
                        )
                    )
                    continue
                latest = sections[-1]
                merged = latest.execute_text + ("\n\n" if latest.execute_text else "") + text
                sections[-1] = SteeringSection(
                    steer_text=latest.steer_text,
                    execute_text=merged.strip(),
                )
        if sections:
            return sections, None
    except element_tree.ParseError as exc:
        parse_warning = f"XML parse warning: {exc}"
        return parse_with_regex_fallback(cleaned=cleaned), parse_warning

    fallback_sections = parse_with_regex_fallback(cleaned=cleaned)
    return fallback_sections, None


def parse_with_regex_fallback(cleaned: str) -> list[SteeringSection]:
    """Fallback parser for malformed XML.

    Args:
        cleaned: Fence-free think text.

    Returns:
        Parsed sections or one raw-content section.
    """
    sections: list[SteeringSection] = []
    for match in TAG_BLOCK_PATTERN.finditer(cleaned):
        tag = match.group(1).lower()
        text = match.group(2).strip()
        if tag in {"steer", "steering"}:
            sections.append(SteeringSection(steer_text=text, execute_text=""))
            continue
        if tag not in {"exec", "execute", "execution"}:
            continue
        if not sections:
            sections.append(
                SteeringSection(
                    steer_text="Unlabeled execution",
                    execute_text=text,
                )
            )
            continue
        latest = sections[-1]
        merged = latest.execute_text + ("\n\n" if latest.execute_text else "") + text
        sections[-1] = SteeringSection(steer_text=latest.steer_text, execute_text=merged)

    if sections:
        return sections
    return [SteeringSection(steer_text="Raw content", execute_text=cleaned)]


def get_first_user_message(messages: list[dict[str, object]]) -> str:
    """Return first user message for context.

    Args:
        messages: Row messages.

    Returns:
        First user content or empty string.
    """
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


def render_section_view(sections: list[SteeringSection]) -> None:
    """Render sections with collapsed execution blocks.

    Args:
        sections: Parsed steering sections.
    """
    for index, section in enumerate(sections, start=1):
        with st.expander(section.label(index=index), expanded=False):
            execution = section.execute_text.strip() or "(empty execution)"
            st.text(execution)


def main() -> None:
    """Run the Streamlit app for transformed-reasoning inspection."""
    st.set_page_config(page_title="Steering Viewer", layout="wide")
    st.title("Steering/Execution Viewer")

    base_dir = Path(__file__).resolve().parent
    candidate_files = discover_dataset_files(base_dir=base_dir)
    if not candidate_files:
        st.error("No transformed JSONL files found.")
        return

    with st.sidebar:
        st.header("Sample Selection")
        selected_file = st.selectbox(
            label="Dataset file",
            options=candidate_files,
            format_func=lambda path: str(path.relative_to(base_dir)),
        )

    rows = load_jsonl_rows(path_str=str(selected_file))
    row_options = build_row_options(rows=rows)
    if not row_options:
        st.error("No rows with `<think>` blocks found in selected file.")
        return

    with st.sidebar:
        selected_row_option = st.selectbox(
            label="Row",
            options=row_options,
            format_func=lambda row_option: row_option.label(),
        )
        selected_row = rows[selected_row_option.row_index]
        messages = selected_row.get("messages")
        assert isinstance(messages, list)
        think_refs = extract_think_refs(messages=messages)
        selected_think_ref = st.selectbox(
            label="Think block",
            options=think_refs,
            format_func=lambda ref: ref.label(),
        )

    sections, parse_warning = parse_xml_sections(xml_text=selected_think_ref.block_text)

    st.subheader("Row Metadata")
    st.json(
        {
            "row_index": selected_row_option.row_index,
            "id": selected_row_option.row_id,
            "dataset_source": selected_row_option.dataset_source,
            "think_blocks_in_row": selected_row_option.think_count,
            "selected_message_index": selected_think_ref.message_index,
            "selected_think_index": selected_think_ref.block_index,
            "sections_parsed": len(sections),
        }
    )

    user_message = get_first_user_message(messages=messages)
    if user_message:
        with st.expander("User Prompt", expanded=False):
            st.text(user_message)

    if parse_warning:
        st.warning(parse_warning)

    st.subheader("Steering Sections")
    render_section_view(sections=sections)

    with st.expander("Raw Think XML", expanded=False):
        st.text(strip_code_fences(text=selected_think_ref.block_text))


if __name__ == "__main__":
    main()
