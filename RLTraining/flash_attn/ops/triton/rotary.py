"""Pure PyTorch fallback for `flash_attn.ops.triton.rotary`."""

from __future__ import annotations

import torch


def _apply_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


def _apply_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: int | torch.Tensor = 0,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """Apply rotary embeddings with the FlashAttention-compatible signature."""

    _ = seqlen_offsets, cu_seqlens, max_seqlen
    if conjugate:
        sin = -sin
    rotary_dim = cos.shape[-1] * 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    cos = cos.unsqueeze(-2).to(dtype=x.dtype, device=x.device)
    sin = sin.unsqueeze(-2).to(dtype=x.dtype, device=x.device)
    output_rot = (
        _apply_interleaved(x=x_rot, cos=cos, sin=sin)
        if interleaved
        else _apply_neox(x=x_rot, cos=cos, sin=sin)
    )
    output = torch.cat((output_rot, x_pass), dim=-1) if x_pass.numel() else output_rot
    if inplace:
        x.copy_(output)
        return x
    return output
