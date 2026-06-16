"""Compatibility exports for `flash_attn.bert_padding`."""

from __future__ import annotations

from einops import rearrange
from verl.utils.npu_flash_attn_utils import (  # pyright: ignore[reportMissingImports]
    index_first_axis,
    pad_input,
    unpad_input,
)

__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
