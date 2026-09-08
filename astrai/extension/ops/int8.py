"""Stateless W8A8 inference primitives with CUDA and CPU reference paths."""

import torch
from torch.library import custom_op

from astrai.extension.loader import get_module

_QMAX = 127.0


def _quantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(x.float() / scale), -127, 127).to(torch.int8)


@custom_op("custom::int8_quantize_dynamic", mutates_args=())
def _quantize_dynamic(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@_quantize_dynamic.register_fake
def _(x):
    return (
        torch.empty_like(x, dtype=torch.int8),
        torch.empty((1,), device=x.device, dtype=torch.float32),
    )


@_quantize_dynamic.register_kernel("cuda")
def _(x):
    if x.dtype != torch.bfloat16:
        raise TypeError(f"dynamic INT8 quantization requires bf16, got {x.dtype}")
    return get_module("int8_ops").quantize_dynamic_bf16(x)


@_quantize_dynamic.register_kernel("cpu")
def _(x):
    scale = (x.abs().amax().float() / _QMAX).clamp_min(1e-12).reshape(1)
    return _quantize(x, scale), scale


@custom_op("custom::int8_quantize_weight", mutates_args=())
def _quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@_quantize_weight.register_fake
def _(w):
    return (
        torch.empty_like(w, dtype=torch.int8),
        torch.empty((w.size(0),), device=w.device, dtype=torch.float32),
    )


@_quantize_weight.register_kernel("cuda")
def _(w):
    if w.dtype != torch.bfloat16 or w.dim() != 2:
        raise TypeError("INT8 weight quantization requires bf16 [N, K]")
    return get_module("int8_ops").quantize_weight_bf16(w)


@_quantize_weight.register_kernel("cpu")
def _(w):
    if w.dim() != 2:
        raise ValueError("w must have shape [N, K]")
    scale = (w.abs().amax(dim=1).float() / _QMAX).clamp_min(1e-12)
    return _quantize(w, scale[:, None]), scale


@custom_op("custom::int8_gemm", mutates_args=())
def _mm_int8(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    pass


@_mm_int8.register_fake
def _(a, b, scale_a, scale_b, bias=None):
    return torch.empty((a.size(0), b.size(0)), device=a.device, dtype=torch.bfloat16)


@_mm_int8.register_kernel("cuda")
def _(a, b, scale_a, scale_b, bias=None):
    return get_module("int8_ops").mm_int8(a, b, scale_a, scale_b, bias)


@_mm_int8.register_kernel("cpu")
def _(a, b, scale_a, scale_b, bias=None):
    out = (a.to(torch.int32) @ b.to(torch.int32).transpose(0, 1)).float()
    out = out * scale_a.float().reshape(1, 1) * scale_b.float().reshape(1, -1)
    if bias is not None and bias.numel():
        out = out + bias.float().reshape(1, -1)
    return out.to(torch.bfloat16)


def quantize_dynamic_bf16(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 activations with one dynamic symmetric scale."""
    if x.dtype != torch.bfloat16:
        raise TypeError(f"x must be bfloat16, got {x.dtype}")
    return _quantize_dynamic(x)


def quantize_weight_bf16(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 [N,K] weights with one scale per output channel."""
    if w.dtype != torch.bfloat16 or w.dim() != 2:
        raise TypeError("w must be a bfloat16 [N, K] tensor")
    return _quantize_weight(w)


def mm_int8(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor,
            scale_b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``a @ b.T`` for INT8 [M,K]/[N,K] operands and return BF16."""
    if a.dtype != torch.int8 or b.dtype != torch.int8 or a.dim() != 2 or b.dim() != 2:
        raise TypeError("a and b must be int8 rank-2 tensors")
    if a.size(1) != b.size(1) or scale_a.numel() != 1 or scale_b.numel() != b.size(0):
        raise ValueError("invalid INT8 GEMM shapes or scales")
    return _mm_int8(a, b, scale_a, scale_b, bias)


def linear_forward_int8(x: torch.Tensor, w_q: torch.Tensor,
                        scale_w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """BF16 linear with static INT8 [N,K] weights and dynamic activation scale."""
    if x.dtype != torch.bfloat16 or w_q.dtype != torch.int8:
        raise TypeError("x must be bf16 and w_q must be int8")
    if x.size(-1) != w_q.size(1):
        raise ValueError("x and w_q inner dimensions must match")
    if x.is_cuda:
        return get_module("int8_ops").linear_forward_int8(x, w_q, scale_w, bias)
    x_q, scale_x = quantize_dynamic_bf16(x.reshape(-1, x.size(-1)))
    out = mm_int8(x_q, w_q, scale_x, scale_w, bias)
    return out.reshape(*x.shape[:-1], w_q.size(0))


__all__ = ["quantize_dynamic_bf16", "quantize_weight_bf16", "mm_int8", "linear_forward_int8"]
