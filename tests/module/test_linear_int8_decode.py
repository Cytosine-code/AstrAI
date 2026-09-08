import torch
import torch.nn.functional as F

from astrai.extension.ops.int8 import linear_forward_int8
from astrai.model.components.linear import Linear
from tests.conftest import skip_no_int8


def test_int8_decode_switch_keeps_cpu_linear_on_bf16_fallback():
    layer = Linear(4, 5, bias=True).to(dtype=torch.bfloat16).eval()
    x = torch.randn(1, 4, dtype=torch.bfloat16)

    assert not layer.set_int8_decode_enabled(True)
    with torch.no_grad():
        actual = layer(x)
        expected = F.linear(x, layer.weight, layer.bias)

    torch.testing.assert_close(actual, expected)
    assert layer._int8_weight is None
    assert layer._int8_scale is None


def test_int8_decode_switch_clears_inference_cache():
    layer = Linear(4, 5)
    layer._int8_weight = torch.empty((5, 4), dtype=torch.int8)
    layer._int8_scale = torch.ones(5, dtype=torch.float32)
    layer._int8_weight_version = 7

    assert not layer.set_int8_decode_enabled(False)
    assert layer._int8_weight is None
    assert layer._int8_scale is None
    assert layer._int8_weight_version == -1
    assert not layer._int8_weights_external


def test_lm_head_role_allows_int8_for_large_vocab_shape():
    layer = Linear(4, 100000, int8_decode_supported=True)
    assert layer.int8_decode_eligible


def test_prequantized_weight_is_attached_without_requantization():
    layer = Linear(4, 5, int8_decode_supported=True)
    weight_q = torch.ones((5, 4), dtype=torch.int8)
    scale_w = torch.full((5,), 0.25, dtype=torch.float32)

    assert layer.set_int8_weights(weight_q, scale_w)
    assert layer._int8_decode_enabled
    assert layer._int8_weights_external
    assert torch.equal(layer._int8_weight, weight_q)
    assert torch.equal(layer._int8_scale, scale_w)


def test_attention_shape_stays_disabled_until_shared_quantization_exists():
    layer = Linear(1536, 1536)
    assert not layer.int8_decode_eligible


@skip_no_int8
def test_int8_decode_uses_cached_weight_for_supported_single_token_shape():
    torch.manual_seed(11)
    layer = Linear(
        1536, 6912, int8_decode_supported=True
    ).to(device="cuda", dtype=torch.bfloat16).eval()
    x = torch.randn((1, 1536), device="cuda", dtype=torch.bfloat16)

    assert layer.set_int8_decode_enabled(True)
    assert layer._int8_weight is not None
    assert layer._int8_scale is not None
    with torch.no_grad():
        actual = layer(x)
        expected = linear_forward_int8(x, layer._int8_weight, layer._int8_scale)

    torch.testing.assert_close(actual, expected)
