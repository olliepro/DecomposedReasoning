from __future__ import annotations

import json
import re
import xml.etree.ElementTree as element_tree
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

THINK_BLOCK_PATTERN = re.compile(
    r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE
)
TAG_BLOCK_PATTERN = re.compile(
    r"<(steer|steering|exec|execute|execution)>(.*?)</\1>",
    flags=re.DOTALL | re.IGNORECASE,
)
STEER_TAG_PATTERN = re.compile(r"<(?:steer|steering)\b[^>]*>", flags=re.IGNORECASE)


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
        think_tokens: Think token count for row (exact if available, else estimate).
        steer_instance_count: Number of steer instances across row think blocks.
        think_to_steer_instances_ratio: Think-to-steer ratio for row.
    """

    row_index: int
    row_id: str
    dataset_source: str
    think_count: int
    think_tokens: int
    steer_instance_count: int
    think_to_steer_instances_ratio: float | None

    def label(self) -> str:
        """Build row selector label.

        Returns:
            Human-readable row label.
        """
        ratio_text = (
            f"{self.think_to_steer_instances_ratio:.2f}"
            if self.think_to_steer_instances_ratio is not None
            else "n/a"
        )
        return (
            f"row {self.row_index} | id={self.row_id} | "
            f"source={self.dataset_source} | think={self.think_count} | "
            f"ratio={ratio_text}"
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


def discover_think_token_stats_path(base_dir: Path, dataset_path: Path) -> Path | None:
    """Return nearest think-token-stats parquet for selected dataset.

    Args:
        base_dir: BuildSFTDataset directory.
        dataset_path: Selected transformed JSONL path.

    Returns:
        Path to `think_token_stats.parquet` if found, else None.
    """

    candidate_paths: list[Path] = []
    for parent in [dataset_path.parent, *dataset_path.parents]:
        candidate_paths.append(
            parent / "cluster_analysis" / "think_token_stats.parquet"
        )
    candidate_paths.append(
        base_dir / "output" / "cluster_analysis" / "think_token_stats.parquet"
    )
    candidate_paths.append(
        base_dir / "output_nonbatch5" / "cluster_analysis" / "think_token_stats.parquet"
    )
    seen: set[Path] = set()
    for candidate_path in candidate_paths:
        resolved_candidate = candidate_path.resolve()
        if resolved_candidate in seen:
            continue
        seen.add(resolved_candidate)
        if candidate_path.exists():
            return candidate_path
    return None


@cache_data_if_available
def load_think_token_map(path_str: str) -> dict[str, int]:
    """Load row_id -> new think token count from parquet.

    Args:
        path_str: Path to `think_token_stats.parquet`.

    Returns:
        Mapping of row id to integer think-token count.
    """

    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return {}

    stats_path = Path(path_str)
    if not stats_path.exists():
        return {}
    stats_df = pd.read_parquet(stats_path)
    required_columns = {"row_id", "new_think_tokens"}
    if not required_columns.issubset(stats_df.columns):
        return {}

    token_map: dict[str, int] = {}
    for row_id, token_value in zip(
        stats_df["row_id"].astype(str).tolist(),
        stats_df["new_think_tokens"].astype(int).tolist(),
    ):
        if row_id not in token_map:
            token_map[row_id] = int(token_value)
    return token_map


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


def count_steer_instances(text: str) -> int:
    """Count steer/steering tags in one think block.

    Args:
        text: Think block text.

    Returns:
        Number of steer instances.
    """

    return len(STEER_TAG_PATTERN.findall(text))


def estimate_think_tokens(think_refs: list[ThinkBlockRef]) -> int:
    """Estimate think-token count from whitespace tokens.

    Args:
        think_refs: Think block references from one row.

    Returns:
        Approximate token count.
    """

    return sum(len(ref.block_text.split()) for ref in think_refs)


def compute_think_to_steer_ratio(
    think_tokens: int, steer_instance_count: int
) -> float | None:
    """Compute think-to-steer-instance ratio.

    Args:
        think_tokens: Think token count.
        steer_instance_count: Number of steer instances.

    Returns:
        Ratio or None when steer count is zero.
    """

    if steer_instance_count <= 0:
        return None
    return float(think_tokens) / float(steer_instance_count)


def build_row_options(
    rows: list[dict[str, object]],
    *,
    think_token_map: dict[str, int] | None = None,
) -> list[RowOption]:
    """Build row options that contain at least one think block.

    Args:
        rows: Parsed dataset rows.
        think_token_map: Optional row-id map of exact think-token counts.

    Returns:
        Selectable row options.
    """
    token_map = think_token_map or {}
    options: list[RowOption] = []
    for row_index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        think_refs = extract_think_refs(messages=messages)
        think_count = len(think_refs)
        if think_count == 0:
            continue
        row_id = str(row.get("id", f"row-{row_index}"))
        steer_instance_count = sum(
            count_steer_instances(text=ref.block_text) for ref in think_refs
        )
        think_tokens = token_map.get(
            row_id, estimate_think_tokens(think_refs=think_refs)
        )
        think_to_steer_instances_ratio = compute_think_to_steer_ratio(
            think_tokens=think_tokens,
            steer_instance_count=steer_instance_count,
        )
        options.append(
            RowOption(
                row_index=row_index,
                row_id=row_id,
                dataset_source=str(row.get("dataset_source", "unknown")),
                think_count=think_count,
                think_tokens=think_tokens,
                steer_instance_count=steer_instance_count,
                think_to_steer_instances_ratio=think_to_steer_instances_ratio,
            )
        )
    return options


def sort_row_options(row_options: list[RowOption], sort_mode: str) -> list[RowOption]:
    """Sort row options by selected mode.

    Args:
        row_options: Row options.
        sort_mode: Sort mode label.

    Returns:
        Sorted options.
    """

    if sort_mode == "Lowest think/steer ratio":
        return sorted(
            row_options,
            key=lambda option: (
                (
                    float("inf")
                    if option.think_to_steer_instances_ratio is None
                    else option.think_to_steer_instances_ratio
                ),
                option.row_index,
            ),
        )
    if sort_mode == "Highest think/steer ratio":
        return sorted(
            row_options,
            key=lambda option: (
                (
                    float("-inf")
                    if option.think_to_steer_instances_ratio is None
                    else option.think_to_steer_instances_ratio
                ),
                -option.row_index,
            ),
            reverse=True,
        )
    if sort_mode == "Most steer instances":
        return sorted(
            row_options,
            key=lambda option: (option.steer_instance_count, -option.row_index),
            reverse=True,
        )
    return sorted(row_options, key=lambda option: option.row_index)


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
                merged = (
                    latest.execute_text + ("\n\n" if latest.execute_text else "") + text
                )
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
        sections[-1] = SteeringSection(
            steer_text=latest.steer_text, execute_text=merged
        )

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
    think_token_stats_path = discover_think_token_stats_path(
        base_dir=base_dir, dataset_path=selected_file
    )
    think_token_map = (
        load_think_token_map(path_str=str(think_token_stats_path))
        if think_token_stats_path is not None
        else {}
    )
    row_options = build_row_options(rows=rows, think_token_map=think_token_map)
    if not row_options:
        st.error("No rows with `<think>` blocks found in selected file.")
        return

    with st.sidebar:
        sort_mode = st.selectbox(
            label="Row sort",
            options=[
                "Dataset order",
                "Lowest think/steer ratio",
                "Highest think/steer ratio",
                "Most steer instances",
            ],
        )
        row_options = sort_row_options(row_options=row_options, sort_mode=sort_mode)
        if think_token_stats_path is None or not think_token_map:
            st.caption(
                "Using estimated think tokens from text splitting (stats file not found)."
            )
        else:
            stats_path_label = (
                str(think_token_stats_path.relative_to(base_dir))
                if think_token_stats_path.is_relative_to(base_dir)
                else str(think_token_stats_path)
            )
            st.caption("Using exact think token counts from: " f"{stats_path_label}")
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
            "think_tokens": selected_row_option.think_tokens,
            "steer_instances": selected_row_option.steer_instance_count,
            "think_to_steer_instances_ratio": (
                selected_row_option.think_to_steer_instances_ratio
            ),
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
