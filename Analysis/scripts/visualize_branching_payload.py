"""Compatibility wrapper for compact branching doc payload builders."""

from __future__ import annotations

try:
    from scripts.visualize_branching_doc_data import (
        node_detail_payload_for_attempt,
        node_detail_payloads_for_attempt,
        tree_payload_for_attempt,
    )
except ModuleNotFoundError:
    from visualize_branching_doc_data import (
        node_detail_payload_for_attempt,
        node_detail_payloads_for_attempt,
        tree_payload_for_attempt,
    )

__all__ = [
    "tree_payload_for_attempt",
    "node_detail_payloads_for_attempt",
    "node_detail_payload_for_attempt",
]
