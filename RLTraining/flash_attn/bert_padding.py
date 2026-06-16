"""Compatibility exports for `flash_attn.bert_padding`.

Verl's CUDA path imports these padding utilities directly from FlashAttention.
The implementations already live in Verl's backend-neutral NPU helper and are
pure PyTorch/einops, which is enough for the Qwen3.5 smoke path.
"""

from __future__ import annotations

from einops import rearrange
from verl.utils.npu_flash_attn_utils import (  # pyright: ignore[reportMissingImports]
    index_first_axis,
    pad_input,
    unpad_input,
)

__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
