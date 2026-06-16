"""Custom math reward function for branching DAPO runs."""

import re
from dataclasses import dataclass
from typing import Any, Union, cast

from branching_dapo.bootstrap import ensure_repo_paths
from branching_dapo.config_types import BranchAdvantageIndex
from math_verify import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    LatexNormalizationConfig,
    StringExtractionConfig,
    parse,
    verify,
)
from math_verify.parser import ExtractionTarget
from sympy import Basic, MatrixBase

ensure_repo_paths()

PREDICTION_WINDOW_CHARS = 300
# `verl` computes reward scores inside a thread executor. Math-Verify's
# signal-based timeout is incompatible with that environment, so we disable the
# internal timeout here and rely on the outer worker lifecycle instead.
PARSE_TIMEOUT_SECONDS = 0
VERIFY_TIMEOUT_SECONDS = 0
ParsedMathValue = Union[Basic, MatrixBase, str]
THINK_BLOCK_PATTERN = re.compile(
    r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL
)
THINK_OPEN_PATTERN = re.compile(r"<think>", flags=re.IGNORECASE)
THINK_CLOSE_PATTERN = re.compile(r"</think>", flags=re.IGNORECASE)
STEER_EXEC_TAG_PATTERN = re.compile(r"</?(?:steer|exec)>", flags=re.IGNORECASE)
STEER_EXEC_SEGMENT_PATTERN = re.compile(
    r"<(?P<tag>steer|exec)>(?P<content>.*?)</(?P=tag)>",
    flags=re.IGNORECASE | re.DOTALL,
)
STEER_BLOCK_PATTERN = re.compile(
    r"<steer>(?P<content>.*?)(?:</steer>|$)",
    flags=re.IGNORECASE | re.DOTALL,
)
EXEC_BLOCK_PATTERN = re.compile(
    r"<exec>(?P<content>.*?)(?:</exec>|$)",
    flags=re.IGNORECASE | re.DOTALL,
)
WORD_PATTERN = re.compile(r"\S+")
BOXED_OPEN_TOKEN = r"\boxed{"
TEXT_WRAPPER_PATTERN = re.compile(r"^\\(?:boxed|text|mathrm|operatorname)\{(.+)\}$")
PREDICTION_EXTRACTION_CONFIG: tuple[ExtractionTarget, ...] = (
    LatexExtractionConfig(
        boxed_match_priority=0,
        normalization_config=LatexNormalizationConfig(
            basic_latex=True,
            units=True,
            malformed_operators=True,
            nits=True,
            boxed="all",
            equations=False,
        ),
    ),
    ExprExtractionConfig(),
)
GROUND_TRUTH_EXTRACTION_CONFIG: tuple[ExtractionTarget, ...] = (
    LatexExtractionConfig(),
    ExprExtractionConfig(),
)


@dataclass(frozen=True)
class ParsedAnswer:
    """Container for one extracted answer candidate sequence.

    Args:
        values: Parsed answer candidates produced by `Math-Verify`.
        parse_mode: Extraction mode used to produce `values`.

    Returns:
        Parsed answer metadata used by the reward function.
    """

    values: tuple[ParsedMathValue, ...]
    parse_mode: str

    def first_value_str(self) -> str | None:
        """Return the first extracted value as a display string.

        Args:
            None.

        Returns:
            First extracted value rendered as a string, or `None`.
        """

        if not self.values:
            return None
        return str(self.values[0])


@dataclass(frozen=True)
class ResponseStructureValidation:
    """Validation result for the assistant response format.

    Args:
        answer_text: Text after the `<think>` block used for answer extraction.
        boxed_answer: Complete boxed expression after the think block, if present.
        steer_exec_present: Whether the think block contains steer/exec tags.
        steer_exec_pair_count: Number of complete steer/exec pairs.
        require_steer_exec: Whether missing steer/exec pairs are a structure error.
        issues: Stable validation issue labels.

    Returns:
        Structured response-format validation metadata.
    """

    answer_text: str
    boxed_answer: str | None
    steer_exec_present: bool
    steer_exec_pair_count: int
    require_steer_exec: bool
    issues: tuple[str, ...]

    def is_valid(self) -> bool:
        """Return whether the response satisfies all structure requirements.

        Args:
            None.

        Returns:
            `True` when no structure issues were recorded.
        """

        return not self.issues

    def has_boxed_output(self) -> bool:
        """Return whether a complete boxed answer is present outside think text.

        Args:
            None.

        Returns:
            `True` when a complete boxed expression was found.
        """

        return self.boxed_answer is not None


@dataclass(frozen=True)
class SteerExecBlockStats:
    """Word-count stats for generated steer and exec blocks."""

    exec_block_count: int
    exec_block_word_count: int
    steer_block_count: int
    steer_block_word_count: int


@dataclass(frozen=True)
class RepeatMetadata:
    """Repeat-loop metadata forwarded from rollout execution."""

    stop_reason: str | None
    block_kind: str | None
    block_count: int
    last_similarity_ratio: float | None

    def forced_close(self) -> bool:
        """Return whether rollout execution forced a repeat close."""

        return self.stop_reason is not None

    def is_kind(self, *, block_kind: str) -> bool:
        """Return whether the repeated block family matches `block_kind`."""

        return self.block_kind == block_kind


def optional_text_value(*, value: object) -> str | None:
    """Return a non-empty string value, or `None`."""

    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def optional_int_value(*, value: object) -> int | None:
    """Return an integer value when coercion is possible."""

    if value is None:
        return None
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def optional_float_value(*, value: object) -> float | None:
    """Return a float value when coercion is possible."""

    if value is None:
        return None
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return None


def repeat_metadata_from_payload(*, rollout_payload: object) -> RepeatMetadata:
    """Extract repeat-loop metadata from rollout reward scores."""

    if not isinstance(rollout_payload, dict):
        return RepeatMetadata(None, None, 0, None)
    leaf_runtime = rollout_payload.get("leaf_runtime", {})
    fallback = leaf_runtime if isinstance(leaf_runtime, dict) else {}
    stop_reason = optional_text_value(
        value=rollout_payload.get(
            "repeat_stop_reason", fallback.get("repeat_stop_reason")
        )
    )
    block_kind = optional_text_value(
        value=rollout_payload.get(
            "repeat_block_kind", fallback.get("repeat_block_kind")
        )
    )
    block_count = optional_int_value(
        value=rollout_payload.get(
            "repeat_block_count", fallback.get("repeat_block_count")
        )
    )
    similarity = optional_float_value(
        value=rollout_payload.get(
            "repeat_last_similarity_ratio",
            fallback.get("repeat_last_similarity_ratio"),
        )
    )
    return RepeatMetadata(stop_reason, block_kind, block_count or 0, similarity)


def normalize_ground_truth(ground_truth: object) -> str:
    """Normalize raw reward-model ground truth into a single math answer string.

    Args:
        ground_truth: Raw reward-model ground truth payload.

    Returns:
        One normalized ground-truth string.
    """

    if isinstance(ground_truth, str):
        return ground_truth
    if isinstance(ground_truth, list):
        values = [str(item) for item in ground_truth if str(item).strip()]
        assert values, "ground_truth list must contain at least one non-empty value"
        return values[0]
    raise TypeError(f"Unsupported ground_truth type: {type(ground_truth).__name__}")


def dedupe_issues(*, issues: list[str]) -> tuple[str, ...]:
    """Deduplicate validation issues while preserving first-seen order.

    Args:
        issues: Issue labels in discovery order.

    Returns:
        Stable tuple of deduplicated issue labels.
    """

    seen: set[str] = set()
    deduped: list[str] = []
    for issue in issues:
        if issue in seen:
            continue
        seen.add(issue)
        deduped.append(issue)
    return tuple(deduped)


def coerce_bool(*, value: bool | str) -> bool:
    """Normalize booleans forwarded through Hydra reward kwargs."""

    if isinstance(value, bool):
        return value
    normalized_value = value.strip().lower()
    if normalized_value in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean-like value, got {value!r}")


def block_word_counts(*, text: str, pattern: re.Pattern[str]) -> tuple[int, ...]:
    """Return whitespace-token word counts for matched block contents."""

    return tuple(
        len(WORD_PATTERN.findall(str(match.group("content"))))
        for match in pattern.finditer(text)
    )


def compute_steer_exec_block_stats(*, solution_str: str) -> SteerExecBlockStats:
    """Count generated steer/exec blocks and their whitespace-token lengths."""

    exec_block_word_counts = block_word_counts(
        text=solution_str, pattern=EXEC_BLOCK_PATTERN
    )
    steer_block_word_counts = block_word_counts(
        text=solution_str, pattern=STEER_BLOCK_PATTERN
    )
    return SteerExecBlockStats(
        exec_block_count=len(exec_block_word_counts),
        exec_block_word_count=sum(exec_block_word_counts),
        steer_block_count=len(steer_block_word_counts),
        steer_block_word_count=sum(steer_block_word_counts),
    )


def extract_complete_boxed_answers(*, text: str) -> tuple[str, ...]:
    """Return all complete `\\boxed{...}` expressions in the text.

    Args:
        text: Raw text to scan.

    Returns:
        Complete boxed expressions in discovery order.
    """

    matches: list[str] = []
    cursor = 0
    while True:
        start = text.find(BOXED_OPEN_TOKEN, cursor)
        if start < 0:
            return tuple(matches)
        depth = 1
        index = start + len(BOXED_OPEN_TOKEN)
        while index < len(text):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    matches.append(text[start : index + 1])
                    break
            index += 1
        cursor = start + 1


def extract_last_complete_boxed_answer(*, text: str) -> str | None:
    """Return the last complete `\\boxed{...}` expression in the text."""

    boxed_answers = extract_complete_boxed_answers(text=text)
    if not boxed_answers:
        return None
    return boxed_answers[-1]


def validate_steer_exec_sequence(
    *, think_text: str
) -> tuple[bool, int, tuple[str, ...]]:
    """Validate optional `<steer>/<exec>` pairs inside one think block.

    Args:
        think_text: Inner think-block text.

    Returns:
        Tuple of presence flag, paired section count, and issue labels.
    """

    issues: list[str] = []
    segment_matches = list(STEER_EXEC_SEGMENT_PATTERN.finditer(think_text))
    segment_tags_present = bool(STEER_EXEC_TAG_PATTERN.search(think_text))
    if not segment_tags_present:
        return False, 0, ()
    steer_count = sum(
        1 for match in segment_matches if str(match.group("tag")).lower() == "steer"
    )
    exec_count = sum(
        1 for match in segment_matches if str(match.group("tag")).lower() == "exec"
    )
    cursor = 0
    for segment_index, match in enumerate(segment_matches):
        if think_text[cursor : match.start()].strip():
            issues.append("non_whitespace_outside_steer_exec")
        expected_tag = "steer" if segment_index % 2 == 0 else "exec"
        actual_tag = str(match.group("tag")).lower()
        if actual_tag != expected_tag:
            issues.append("steer_exec_not_interleaved")
        if not str(match.group("content")).strip():
            issues.append("empty_steer_exec_block")
        cursor = match.end()
    if think_text[cursor:].strip():
        issues.append("non_whitespace_outside_steer_exec")
    if not segment_matches:
        issues.append("incomplete_steer_exec_tags")
    if steer_count != exec_count:
        issues.append("unequal_steer_exec_block_counts")
    if len(segment_matches) % 2 != 0:
        issues.append("unpaired_steer_exec_tags")
    if segment_matches and str(segment_matches[0].group("tag")).lower() != "steer":
        issues.append("steer_exec_must_start_with_steer")
    paired_section_count = min(steer_count, exec_count)
    return True, paired_section_count, dedupe_issues(issues=issues)


def validate_response_structure(
    *, solution_str: str, require_steer_exec: bool | str = True
) -> ResponseStructureValidation:
    """Validate think/steer/exec/boxed formatting for one assistant response.

    Args:
        solution_str: Full assistant response text.
        require_steer_exec: Whether valid think text must contain steer/exec pairs.

    Returns:
        Structured validation result for reward gating.
    """

    require_steer_exec_bool = coerce_bool(value=require_steer_exec)
    issues: list[str] = []
    think_matches = list(THINK_BLOCK_PATTERN.finditer(solution_str))
    if len(think_matches) != 1:
        boxed_answers = extract_complete_boxed_answers(text=solution_str)
        issues.append("expected_single_complete_think_block")
        if not boxed_answers:
            issues.append("missing_boxed_answer")
        return ResponseStructureValidation(
            answer_text=solution_str.strip(),
            boxed_answer=boxed_answers[-1] if boxed_answers else None,
            steer_exec_present=bool(STEER_EXEC_TAG_PATTERN.search(solution_str)),
            steer_exec_pair_count=0,
            require_steer_exec=require_steer_exec_bool,
            issues=dedupe_issues(issues=issues),
        )
    if (
        len(THINK_OPEN_PATTERN.findall(solution_str)) != 1
        or len(THINK_CLOSE_PATTERN.findall(solution_str)) != 1
    ):
        issues.append("expected_single_complete_think_block")
    think_match = think_matches[0]
    think_text = str(think_match.group(1))
    prefix_text = solution_str[: think_match.start()]
    suffix_text = solution_str[think_match.end() :]
    answer_text = suffix_text.strip()
    if prefix_text.strip():
        issues.append("non_whitespace_before_think_block")
    if not think_text.strip():
        issues.append("empty_think_block")
    if STEER_EXEC_TAG_PATTERN.search(prefix_text) or STEER_EXEC_TAG_PATTERN.search(
        suffix_text
    ):
        issues.append("steer_exec_outside_think_block")
    steer_exec_present, steer_exec_pair_count, steer_exec_issues = (
        validate_steer_exec_sequence(think_text=think_text)
    )
    if require_steer_exec_bool and not steer_exec_present:
        issues.append("missing_steer_exec_blocks")
    issues.extend(list(steer_exec_issues))
    boxed_answers = extract_complete_boxed_answers(text=answer_text)
    if not boxed_answers:
        issues.append("missing_boxed_answer")
    return ResponseStructureValidation(
        answer_text=answer_text,
        boxed_answer=boxed_answers[-1] if boxed_answers else None,
        steer_exec_present=steer_exec_present,
        steer_exec_pair_count=steer_exec_pair_count,
        require_steer_exec=require_steer_exec_bool,
        issues=dedupe_issues(issues=issues),
    )


def prediction_tail(*, solution_str: str) -> str:
    """Trim the model response to the suffix most likely to contain the final answer.

    Args:
        solution_str: Full model response text.

    Returns:
        Final response suffix used for answer extraction.
    """

    return solution_str[-PREDICTION_WINDOW_CHARS:].strip()


def unwrap_text_wrappers(*, value: str) -> tuple[str, ...]:
    """Build string-match variants by peeling simple LaTeX wrappers.

    Args:
        value: Raw ground-truth string.

    Returns:
        Deduplicated string variants for targeted string extraction.

    Example:
        >>> unwrap_text_wrappers(value=r"\\boxed{\\text{No}}")
        ('\\\\boxed{\\\\text{No}}', '\\\\text{No}', 'No')
    """

    variants: list[str] = []
    pending = [value.strip()]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if not current or current in seen:
            continue
        seen.add(current)
        variants.append(current)
        match = TEXT_WRAPPER_PATTERN.fullmatch(current)
        if match is not None:
            pending.append(match.group(1).strip())
    return tuple(variants)


def normalize_string_match(*, value: str) -> str:
    """Canonicalize a string-answer candidate for comparison.

    Args:
        value: Raw extracted string candidate.

    Returns:
        Lowercased candidate with whitespace collapsed.
    """

    normalized = value.strip().lower()
    normalized = normalized.replace("\\dfrac", "\\frac")
    normalized = normalized.replace("\\tfrac", "\\frac")
    return " ".join(normalized.split())


def build_string_extraction_config(
    *, ground_truth: str
) -> tuple[ExtractionTarget, ...]:
    """Create the targeted string extraction config for one ground truth answer.

    Args:
        ground_truth: One normalized ground-truth answer string.

    Returns:
        `Math-Verify` extraction config tuned to the ground-truth string variants.
    """

    string_variants = unwrap_text_wrappers(value=ground_truth)
    return (
        StringExtractionConfig(
            strings=string_variants, try_extract_without_anchor=True, lowercase=False
        ),
    )


def parse_answer(
    *,
    answer_text: str,
    extraction_config: tuple[ExtractionTarget, ...],
    parse_mode: str,
) -> ParsedAnswer:
    """Parse one answer string with `Math-Verify`.

    Args:
        answer_text: Raw answer text to parse.
        extraction_config: `Math-Verify` extraction rules.
        parse_mode: Label describing the extraction mode.

    Returns:
        Parsed answer candidates and the extraction mode label.
    """

    parsed_values = parse(
        pred=answer_text,
        extraction_config=extraction_config,
        parsing_timeout=PARSE_TIMEOUT_SECONDS,
        raise_on_error=False,
    )
    return ParsedAnswer(
        values=cast(tuple[ParsedMathValue, ...], tuple(parsed_values)),
        parse_mode=parse_mode,
    )


def parse_ground_truth_answer(*, ground_truth: str) -> ParsedAnswer:
    """Parse ground truth as math first, then as a targeted string fallback.

    Args:
        ground_truth: One normalized ground-truth answer string.

    Returns:
        Parsed ground-truth candidates.
    """

    parsed_ground_truth = parse_answer(
        answer_text=ground_truth,
        extraction_config=GROUND_TRUTH_EXTRACTION_CONFIG,
        parse_mode="math",
    )
    if parsed_ground_truth.values:
        return parsed_ground_truth
    return parse_answer(
        answer_text=ground_truth,
        extraction_config=build_string_extraction_config(ground_truth=ground_truth),
        parse_mode="string",
    )


def parse_prediction_answer(
    *, solution_str: str, ground_truth: str, ground_truth_answer: ParsedAnswer
) -> ParsedAnswer:
    """Parse one model response with a mode aligned to the parsed ground truth.

    Args:
        solution_str: Full model response text.
        ground_truth: One normalized ground-truth answer string.
        ground_truth_answer: Parsed ground-truth candidates.

    Returns:
        Parsed prediction candidates.
    """

    boxed_answer = extract_last_complete_boxed_answer(text=solution_str)
    if boxed_answer is None:
        return ParsedAnswer(values=(), parse_mode=ground_truth_answer.parse_mode)
    candidate_solution = boxed_answer
    if ground_truth_answer.parse_mode == "math":
        return parse_answer(
            answer_text=candidate_solution,
            extraction_config=PREDICTION_EXTRACTION_CONFIG,
            parse_mode="math",
        )
    parsed_prediction = parse_answer(
        answer_text=candidate_solution,
        extraction_config=build_string_extraction_config(ground_truth=ground_truth),
        parse_mode="string",
    )
    if parsed_prediction.values:
        return parsed_prediction
    boxed_variants = unwrap_text_wrappers(value=boxed_answer)
    return ParsedAnswer(
        values=cast(tuple[ParsedMathValue, ...], boxed_variants[1:] or boxed_variants),
        parse_mode="string",
    )


def string_answers_match(
    *, prediction_values: tuple[ParsedMathValue, ...], ground_truth: str
) -> bool:
    """Compare extracted string answers against normalized ground-truth variants.

    Args:
        prediction_values: Extracted prediction candidates from `Math-Verify`.
        ground_truth: One normalized ground-truth answer string.

    Returns:
        `True` when any extracted prediction matches the normalized target strings.
    """

    if not prediction_values:
        return False
    normalized_targets = {
        normalize_string_match(value=item)
        for item in unwrap_text_wrappers(value=ground_truth)
    }
    normalized_predictions = {
        normalize_string_match(value=str(item)) for item in prediction_values
    }
    return bool(normalized_targets & normalized_predictions)


def build_branch_advantage_index(
    data_source: str, extra_info: dict[str, Any]
) -> BranchAdvantageIndex:
    """Build the branch-aware `uid` payload consumed by the custom estimator.

    Args:
        data_source: Exported `data_source` field for this row.
        extra_info: Extra-info payload forwarded by the DAPO reward manager.

    Returns:
        Serialized branch-advantage index dataclass.
    """

    rollout_payload = extra_info.get("rollout_reward_scores", {})
    branch_metadata = (
        rollout_payload.get("branch_metadata", {})
        if isinstance(rollout_payload, dict)
        else {}
    )
    prompt_uid = str(
        branch_metadata.get("prompt_uid", extra_info.get("source_row_id", data_source))
    )
    path_node_ids = branch_metadata.get("path_node_ids", ["node_root"])
    assert isinstance(path_node_ids, list), "path_node_ids must be a list"
    branch_token_offsets = branch_metadata.get("branch_token_offsets", [])
    assert isinstance(branch_token_offsets, list), "branch_token_offsets must be a list"
    leaf_node_id = str(branch_metadata.get("leaf_node_id", path_node_ids[-1]))
    branch_depth = int(
        branch_metadata.get("branch_depth", max(len(path_node_ids) - 1, 0))
    )
    assert len(branch_token_offsets) == branch_depth, (
        "branch_token_offsets must align with branch_depth: "
        f"{len(branch_token_offsets)} != {branch_depth}"
    )
    return BranchAdvantageIndex(
        prompt_uid=prompt_uid,
        branch_tree_id=str(branch_metadata.get("branch_tree_id", f"tree:{prompt_uid}")),
        leaf_id=str(branch_metadata.get("leaf_id", f"leaf:{prompt_uid}")),
        leaf_node_id=leaf_node_id,
        path_node_ids=tuple(str(node_id) for node_id in path_node_ids),
        branch_token_offsets=tuple(int(offset) for offset in branch_token_offsets),
        parent_branch_id=(
            None
            if branch_metadata.get("parent_branch_id") is None
            else str(branch_metadata["parent_branch_id"])
        ),
        branch_depth=branch_depth,
        selected_cluster_id=(
            None
            if branch_metadata.get("selected_cluster_id") is None
            else str(branch_metadata["selected_cluster_id"])
        ),
        cluster_name=(
            None
            if branch_metadata.get("cluster_name") is None
            else str(branch_metadata["cluster_name"])
        ),
        selector_mode=str(branch_metadata.get("selector_mode", "random")),
        candidate_pool_key=(
            None
            if branch_metadata.get("candidate_pool_key") is None
            else str(branch_metadata["candidate_pool_key"])
        ),
    )


def prepend_initial_assistant_prefix(
    *, solution_str: str, extra_info: dict[str, Any]
) -> tuple[str, str]:
    """Restore prompt-prefilled assistant text for reward validation.

    Args:
        solution_str: Decoded sampled response string.
        extra_info: Reward metadata containing rollout reward scores.

    Returns:
        Tuple of `(scored_solution, applied_prefix)`.
    """

    rollout_payload = extra_info.get("rollout_reward_scores", {})
    if not isinstance(rollout_payload, dict):
        return solution_str, ""
    raw_prefix = rollout_payload.get("initial_assistant_prefix", "")
    if not isinstance(raw_prefix, str) or not raw_prefix:
        return solution_str, ""
    if THINK_OPEN_PATTERN.search(solution_str):
        return solution_str, ""
    return f"{raw_prefix}{solution_str}", raw_prefix


def resolve_scored_solution(
    *, solution_str: str, extra_info: dict[str, Any]
) -> tuple[str, str, bool]:
    """Resolve the assistant text that should be scored.

    Args:
        solution_str: Decoded sampled response string.
        extra_info: Reward metadata containing rollout reward scores.

    Returns:
        Tuple of `(scored_solution, applied_prefix, used_logical_response_text)`.
    """

    rollout_payload = extra_info.get("rollout_reward_scores", {})
    if isinstance(rollout_payload, dict):
        logical_response_text = rollout_payload.get("logical_response_text", "")
        if isinstance(logical_response_text, str) and logical_response_text.strip():
            raw_prefix = rollout_payload.get("initial_assistant_prefix", "")
            applied_prefix = raw_prefix if isinstance(raw_prefix, str) else ""
            return logical_response_text, applied_prefix, True
    scored_solution, applied_prefix = prepend_initial_assistant_prefix(
        solution_str=solution_str,
        extra_info=extra_info,
    )
    return scored_solution, applied_prefix, False


def compute_math_reward(
    *, solution_str: str, ground_truth: str, require_steer_exec: bool | str = True
) -> dict[str, object]:
    """Compute one `Math-Verify` reward payload for the response and gold answer.

    Args:
        solution_str: Model response text.
        ground_truth: Normalized ground-truth answer string.
        require_steer_exec: Whether valid think text must contain steer/exec pairs.

    Returns:
        Reward payload containing score, accuracy, and extracted prediction text.

    Example:
        >>> compute_math_reward(
        ...     solution_str=r"<think><steer>Plan.</steer><exec>Check.</exec></think> \\boxed{30000}",
        ...     ground_truth="30,000",
        ... )["score"]
        1.1
    """

    structure_validation = validate_response_structure(
        solution_str=solution_str,
        require_steer_exec=require_steer_exec,
    )
    block_stats = compute_steer_exec_block_stats(solution_str=solution_str)
    parsed_ground_truth = parse_ground_truth_answer(ground_truth=ground_truth)
    parsed_prediction = parse_prediction_answer(
        solution_str=structure_validation.answer_text,
        ground_truth=ground_truth,
        ground_truth_answer=parsed_ground_truth,
    )
    if parsed_ground_truth.parse_mode == "string":
        raw_answer_is_correct = string_answers_match(
            prediction_values=parsed_prediction.values,
            ground_truth=ground_truth,
        )
    else:
        raw_answer_is_correct = bool(parsed_prediction.values) and verify(
            gold=list(parsed_ground_truth.values),
            target=list(parsed_prediction.values),
            strict=True,
            allow_set_relation_comp=True,
            timeout_seconds=VERIFY_TIMEOUT_SECONDS,
            raise_on_error=False,
        )
    answer_is_correct = raw_answer_is_correct
    structure_reward = 0.1 if structure_validation.is_valid() else 0.0
    answer_reward = 1.0 if answer_is_correct else 0.0
    extracted_prediction = parsed_prediction.first_value_str()
    return {
        "score": structure_reward + answer_reward,
        "acc": answer_is_correct,
        "answer_acc": answer_is_correct,
        "raw_answer_acc": raw_answer_is_correct,
        "structure_reward": structure_reward,
        "answer_reward": answer_reward,
        "pred": extracted_prediction,
        "parse_mode": parsed_prediction.parse_mode,
        "format_valid": structure_validation.is_valid(),
        "boxed_answer": structure_validation.boxed_answer,
        "boxed_present": structure_validation.has_boxed_output(),
        "steer_exec_present": structure_validation.steer_exec_present,
        "steer_exec_pair_count": structure_validation.steer_exec_pair_count,
        "exec_block_count": block_stats.exec_block_count,
        "exec_block_word_count": block_stats.exec_block_word_count,
        "steer_block_count": block_stats.steer_block_count,
        "steer_block_word_count": block_stats.steer_block_word_count,
        "require_steer_exec": structure_validation.require_steer_exec,
        "structure_issues": list(structure_validation.issues),
    }


def compute_score_branching_dapo(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: object,
    extra_info: dict[str, Any],
    require_steer_exec: bool | str = True,
) -> dict[str, object]:
    """Compute math reward and emit branch metadata for custom advantage.

    Args:
        data_source: Exported row data source.
        solution_str: Model response string.
        ground_truth: Reward-model ground truth payload.
        extra_info: Extra-info mapping forwarded by the DAPO reward manager.
        require_steer_exec: Whether reward structure requires steer/exec pairs.

    Returns:
        Reward payload containing `score` plus branch-aware metadata.
    """

    normalized_ground_truth = normalize_ground_truth(ground_truth=ground_truth)
    scored_solution_str, applied_prefix, used_logical_response_text = (
        resolve_scored_solution(
            solution_str=solution_str,
            extra_info=extra_info,
        )
    )
    math_result = compute_math_reward(
        solution_str=scored_solution_str,
        ground_truth=normalized_ground_truth,
        require_steer_exec=require_steer_exec,
    )
    branch_index = build_branch_advantage_index(
        data_source=data_source, extra_info=extra_info
    )
    rollout_payload = extra_info.get("rollout_reward_scores", {})
    repeat_metadata = repeat_metadata_from_payload(rollout_payload=rollout_payload)
    score_raw = math_result["score"]
    structure_reward_raw = math_result["structure_reward"]
    answer_reward_raw = math_result["answer_reward"]
    steer_exec_pair_count_raw = math_result["steer_exec_pair_count"]
    exec_block_count_raw = math_result["exec_block_count"]
    exec_block_word_count_raw = math_result["exec_block_word_count"]
    steer_block_count_raw = math_result["steer_block_count"]
    steer_block_word_count_raw = math_result["steer_block_word_count"]
    assert isinstance(score_raw, (float, int)), "math_verify score must be numeric"
    assert isinstance(
        structure_reward_raw, (float, int)
    ), "structure_reward must be numeric"
    assert isinstance(answer_reward_raw, (float, int)), "answer_reward must be numeric"
    assert isinstance(
        steer_exec_pair_count_raw, int
    ), "steer_exec_pair_count must be an int"
    assert isinstance(exec_block_count_raw, int), "exec_block_count must be an int"
    assert isinstance(
        exec_block_word_count_raw, int
    ), "exec_block_word_count must be an int"
    assert isinstance(steer_block_count_raw, int), "steer_block_count must be an int"
    assert isinstance(
        steer_block_word_count_raw, int
    ), "steer_block_word_count must be an int"
    score_value = float(score_raw)
    return {
        "score": score_value,
        "acc": bool(math_result["acc"]),
        "answer_acc": bool(math_result["answer_acc"]),
        "raw_answer_acc": bool(math_result["raw_answer_acc"]),
        "structure_reward": float(structure_reward_raw),
        "answer_reward": float(answer_reward_raw),
        "pred": math_result["pred"],
        "branch_uid": branch_index.to_json(),
        "scorer_name": "math_verify",
        "reward_parse_mode": math_result["parse_mode"],
        "format_valid": bool(math_result["format_valid"]),
        "boxed_present": bool(math_result["boxed_present"]),
        "boxed_answer": math_result["boxed_answer"],
        "steer_exec_present": bool(math_result["steer_exec_present"]),
        "steer_exec_pair_count": steer_exec_pair_count_raw,
        "exec_block_count": exec_block_count_raw,
        "exec_block_word_count": exec_block_word_count_raw,
        "steer_block_count": steer_block_count_raw,
        "steer_block_word_count": steer_block_word_count_raw,
        "repeat_forced_think_close": repeat_metadata.forced_close(),
        "repeat_exec_forced_close": repeat_metadata.is_kind(block_kind="exec"),
        "repeat_steer_forced_close": repeat_metadata.is_kind(block_kind="steer"),
        "repeat_stop_reason": repeat_metadata.stop_reason,
        "repeat_block_kind": repeat_metadata.block_kind,
        "repeat_block_count": repeat_metadata.block_count,
        "repeat_last_similarity_ratio": repeat_metadata.last_similarity_ratio,
        "require_steer_exec": bool(math_result["require_steer_exec"]),
        "structure_issues": list(cast(list[str], math_result["structure_issues"])),
        "source_family": str(extra_info.get("source_family", data_source)),
        "initial_assistant_prefix": applied_prefix,
        "used_logical_response_text": used_logical_response_text,
        "branch_tree_id": branch_index.branch_tree_id,
        "leaf_id": branch_index.leaf_id,
        "branch_depth": float(branch_index.branch_depth),
        "selector_mode": branch_index.selector_mode,
        "selected_cluster_id": branch_index.selected_cluster_id,
        "cluster_name": branch_index.cluster_name,
        "candidate_pool_key": branch_index.candidate_pool_key,
    }
