"""Stateless wrappers around compiled extension kernels."""

from astrai.extension.ops.attention import (
    TensorLayout,
    attn_decode,
    attn_paged_decode,
    attn_paged_prefill,
    attn_prefill,
)
from astrai.extension.ops.rotary import rotary_emb
from astrai.extension.ops.int8 import (
    linear_forward_int8,
    mm_int8,
    quantize_dynamic_bf16,
    quantize_weight_bf16,
)

__all__ = [
    "TensorLayout",
    "attn_decode",
    "attn_paged_decode",
    "attn_paged_prefill",
    "attn_prefill",
    "rotary_emb",
    "quantize_dynamic_bf16",
    "quantize_weight_bf16",
    "mm_int8",
    "linear_forward_int8",
]
