import pytest
import torch

from astrai.extension.ops.int8 import (
    linear_forward_int8,
    mm_int8,
    quantize_dynamic_bf16,
    quantize_weight_bf16,
)
from tests.conftest import skip_no_int8


def _reference(aq, wq, sx, sw, bias=None):
    # PyTorch does not implement CUDA int32 addmm. For CUDA inputs, FP32
    # exactly accumulates these test values (well below its 24-bit integer
    # mantissa limit) and therefore remains an INT32-equivalent reference.
    if aq.is_cuda:
        out = aq.float() @ wq.float().transpose(0, 1)
    else:
        out = (aq.to(torch.int32) @ wq.to(torch.int32).transpose(0, 1)).float()
    out = out * sx.float().reshape(1, 1) * sw.float().reshape(1, -1)
    if bias is not None:
        out = out + bias.float().reshape(1, -1)
    return out.to(torch.bfloat16)


def test_int8_quantization_cpu_reference_and_zero_scale():
    x = torch.tensor([[0.0, 1.0, -1.0, 200.0]], dtype=torch.bfloat16)
    q, scale = quantize_dynamic_bf16(x)
    assert q.dtype == torch.int8 and scale.shape == (1,)
    assert scale.item() == pytest.approx(200.0 / 127.0)
    assert torch.equal(q, torch.tensor([[0, 1, -1, 127]], dtype=torch.int8))

    wq, sw = quantize_weight_bf16(torch.zeros((3, 7), dtype=torch.bfloat16))
    assert torch.equal(wq, torch.zeros_like(wq))
    torch.testing.assert_close(sw, torch.full((3,), 1e-12, dtype=torch.float32))


@pytest.mark.parametrize(("m", "n", "k"), [(16, 8, 32), (17, 9, 33), (64, 128, 64)])
def test_int8_mm_cpu_matches_explicit_int32(m, n, k):
    torch.manual_seed(m + n + k)
    x = torch.randn((m, k), dtype=torch.bfloat16)
    w = torch.randn((n, k), dtype=torch.bfloat16)
    aq, sx = quantize_dynamic_bf16(x)
    wq, sw = quantize_weight_bf16(w)
    out = mm_int8(aq, wq, sx, sw)
    torch.testing.assert_close(out, _reference(aq, wq, sx, sw))


def test_int8_linear_cpu_static_weight_bias_and_leading_dims():
    torch.manual_seed(9)
    x = torch.randn((2, 3, 37), dtype=torch.bfloat16)
    w = torch.randn((13, 37), dtype=torch.bfloat16)
    bias = torch.randn(13, dtype=torch.bfloat16)
    wq, sw = quantize_weight_bf16(w)
    out = linear_forward_int8(x, wq, sw, bias)
    xq, sx = quantize_dynamic_bf16(x.reshape(-1, 37))
    expected = _reference(xq, wq, sx, sw, bias).reshape(2, 3, 13)
    torch.testing.assert_close(out, expected)


@skip_no_int8
@pytest.mark.parametrize(("m", "n", "k"), [(16, 8, 32), (17, 9, 33), (64, 128, 64)])
def test_int8_mma_cuda_matches_explicit_int32(m, n, k):
    torch.manual_seed(m + n + k)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    aq, sx = quantize_dynamic_bf16(x)
    wq, sw = quantize_weight_bf16(w)
    out = mm_int8(aq, wq, sx, sw)
    torch.testing.assert_close(out, _reference(aq, wq, sx, sw), atol=1e-2, rtol=1e-2)


@skip_no_int8
@pytest.mark.parametrize(("m", "n", "k"), [(1, 6912, 1536), (16, 6912, 1536), (128, 1536, 6912)])
def test_int8_linear_cuda_model_shapes(m, n, k):
    torch.manual_seed(m + n + k)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    wq, sw = quantize_weight_bf16(w)
    out = linear_forward_int8(x, wq, sw, bias)
    xq, sx = quantize_dynamic_bf16(x)
    torch.testing.assert_close(out, _reference(xq, wq, sx, sw, bias), atol=1e-2, rtol=1e-2)
