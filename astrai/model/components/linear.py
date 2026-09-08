import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.loader import is_available
from astrai.extension.ops.int8 import linear_forward_int8, quantize_weight_bf16


class Linear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        init_std: float = 0.02,
        int8_decode_supported: bool = False,
        int8_decode_attention: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.init_std = init_std
        self._int8_decode_supported = int8_decode_supported
        self._int8_decode_attention = int8_decode_attention
        self._int8_attention_enabled = False
        # Inference cache only: not serialized and never used by training.
        self.register_buffer("_int8_weight", None, persistent=False)
        self.register_buffer("_int8_scale", None, persistent=False)
        self._int8_decode_enabled = False
        self._int8_weight_version = -1
        self._int8_weights_external = False

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=self.init_std)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def int8_decode_eligible(self) -> bool:
        return self._int8_decode_supported or (
            self._int8_decode_attention and self._int8_attention_enabled
        )

    def set_int8_weights(self, weight_q: Tensor, scale_w: Tensor) -> bool:
        """Attach pre-quantized inference weights without re-quantizing them.

        ``weight_q`` is stored as ``[out_features, in_features]`` INT8 and
        ``scale_w`` contains one FP32 scale per output channel. The caller is
        responsible for placing both tensors on the model device.
        """
        if weight_q.dtype != torch.int8 or weight_q.shape != self.weight.shape:
            raise TypeError(
                "weight_q must be int8 with the same shape as the BF16 weight"
            )
        if (
            scale_w.dtype != torch.float32
            or scale_w.device != weight_q.device
            or scale_w.numel() != self.weight.size(0)
        ):
            raise TypeError("scale_w must be CUDA/CPU float32 [out_features]")
        if weight_q.device != self.weight.device:
            raise ValueError("weight_q must be on the same device as weight")
        if not self.int8_decode_eligible:
            raise ValueError("this Linear module is not eligible for INT8 decode")
        self._int8_decode_enabled = True
        self._int8_weight = weight_q.detach()
        self._int8_scale = scale_w.detach().reshape(-1)
        self._int8_weight_version = self.weight._version
        self._int8_weights_external = True
        return True

    def set_int8_decode_enabled(
        self, enabled: bool, include_attention: bool = False
    ) -> bool:
        """Enable decode-only W8A8 dispatch and eagerly cache static weights.

        Returns whether this projection has a supported shape and was prepared.
        Unsupported modules intentionally remain BF16 fallbacks.
        """
        self._int8_decode_enabled = enabled
        self._int8_attention_enabled = include_attention
        if not enabled:
            self._int8_weight = None
            self._int8_scale = None
            self._int8_weight_version = -1
            self._int8_weights_external = False
            return False
        return self._prepare_int8_weight()

    def _prepare_int8_weight(self) -> bool:
        if (
            not self.int8_decode_eligible
            or not is_available("int8_ops")
        ):
            return False
        if self._int8_weights_external:
            return (
                self._int8_weight is not None
                and self._int8_scale is not None
                and self._int8_weight.dtype == torch.int8
                and self._int8_weight.device == self.weight.device
                and self._int8_weight.shape == self.weight.shape
                and self._int8_scale.dtype == torch.float32
                and self._int8_scale.numel() == self.weight.size(0)
            )
        if (
            self._int8_weight is not None
            and self._int8_scale is not None
            and self._int8_weight_version == self.weight._version
        ):
            return True
        if not self.weight.is_cuda or self.weight.dtype != torch.bfloat16:
            return False
        if (
            self._int8_weight is None
            or self._int8_scale is None
            or self._int8_weight_version != self.weight._version
        ):
            with torch.no_grad():
                self._int8_weight, self._int8_scale = quantize_weight_bf16(self.weight)
            self._int8_weight_version = self.weight._version
            self._int8_weights_external = False
        return True

    def _can_use_int8_decode(self, x: Tensor) -> bool:
        return (
            self._int8_decode_enabled
            and not self.training
            and not torch.is_grad_enabled()
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.numel() == self.weight.size(1)
            and self._prepare_int8_weight()
        )

    def forward(self, x: Tensor) -> Tensor:
        if self._can_use_int8_decode(x):
            return linear_forward_int8(x, self._int8_weight, self._int8_scale, self.bias)
        return F.linear(x, self.weight, self.bias)
