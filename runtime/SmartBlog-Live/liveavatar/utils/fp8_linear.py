import torch
import torch.nn as nn
import math
import logging
import os
logger = logging.getLogger()


def _optional_fp8_quant_compile(func):
    if str(os.getenv("LIVEAVATAR_FP8_QUANT_COMPILE", "1")).strip().lower() in {"0", "false", "no", "off"}:
        logger.info("FP8 quantize torch.compile disabled for %s", getattr(func, "__name__", repr(func)))
        return func
    return torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)(func)


@_optional_fp8_quant_compile
def quant_fp8_tensorwise(input, target_dtype=torch.float8_e4m3fn):
    max_value = torch.finfo(target_dtype).max
    amax_input = torch.max(torch.abs(input)).float()
    input_scale = max_value / torch.clamp(amax_input, min=1e-12)
    input_fp8 = (input * input_scale).clamp(-max_value, max_value).to(target_dtype)
    # Return the dequant scale so that: real = fp8 * scale.
    return input_fp8, input_scale.reciprocal()


@_optional_fp8_quant_compile
def quant_fp8_rowwise(input_2d, target_dtype=torch.float8_e4m3fn):
    """
    Row-wise (per-row) FP8 quantization for 2D matrices.

    This is much more stable than a single tensor-wise scale for transformer-like
    activations, where outliers can otherwise zero-out most values.
    """
    max_value = torch.finfo(target_dtype).max
    amax_row = torch.amax(torch.abs(input_2d), dim=1, keepdim=True).float()  # [M, 1]
    scale = max_value / torch.clamp(amax_row, min=1e-12)  # [M, 1]
    fp8 = (input_2d * scale).clamp(-max_value, max_value).to(target_dtype)
    return fp8, scale.reciprocal()


class FP8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, max_input_val, max_weight_val):
        """
        input: [B, *, in_features]  (可以是 2D 或 3D)
        weight: [out_features, in_features]
        bias: [out_features] 或 None
        """
        prev_shape = input.shape  # 保存原始形状

        # ===== 权重量化 =====
        # We store pre-quantized weights as (weight_fp8_T, weight_scale_b):
        # - weight_fp8_T: [in_features, out_features] float8 (contiguous)
        # - weight_scale_b: [1, out_features] float (contiguous)
        if isinstance(weight, tuple):
            weight_fp8_T, weight_scale_b = weight
            in_feature = weight_fp8_T.shape[0]
            out_feature = weight_fp8_T.shape[1]
        else:
            # Fallback path: quantize weights on the fly (slow).
            in_feature = weight.shape[1]
            out_feature = weight.shape[0]
            w_fp8, w_scale_row = quant_fp8_rowwise(weight, torch.float8_e4m3fn)  # [out, in], [out, 1]
            # Keep the transposed *view* (stride(0)==1). _scaled_mm rowwise mode
            # expects this layout for `b`.
            weight_fp8_T = w_fp8.T
            weight_scale_b = w_scale_row.reshape(1, out_feature).contiguous()

        input_2d = input.view(-1, in_feature)

        # ===== 输入量化 (RowWise) =====
        # Use E5M2 for activations to avoid excessive underflow-to-zero when
        # a row has large outliers (E4M3 min normal is relatively large).
        input_fp8, input_scale_a = quant_fp8_rowwise(input_2d, torch.float8_e5m2)  # [M, K], [M, 1]
        input_scale_a = input_scale_a.contiguous()

        # ===== FP8 matmul =====
        out_2d = torch._scaled_mm(
            input_fp8,
            weight_fp8_T,
            scale_a=input_scale_a,
            scale_b=weight_scale_b,
            bias=bias,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

        # 恢复成原来的 batch/seq 形状
        if isinstance(out_2d, tuple):
            out_2d = out_2d[0]
        out = out_2d.view(*prev_shape[:-1], out_feature)
        # assert not torch.isnan(out).any(), "forward contains NaN!"
        out = out.to(input.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError("no implement for backward")


class FP8ScaleLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float16, device="cuda"):
        super().__init__()
        factory_kwargs = {"dtype": dtype, "device": device}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.bias = None
        self.reset_parameters()
        self.max_input_val = torch.finfo(torch.float8_e4m3fn).max
        self.max_weight_val = torch.finfo(torch.float8_e4m3fn).max
        self.quantized_weight = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def quantize_weight(self,):
        # Use per-output-channel (row-wise) scaling for weights.
        fp8_w, scale_row = quant_fp8_rowwise(self.weight.detach(), torch.float8_e4m3fn)  # [out, in], [out, 1]
        # Keep the transposed *view* (stride(0)==1). _scaled_mm rowwise mode
        # expects this layout for `b`.
        self.register_buffer("weight_fp8_T", fp8_w.T)
        # scaled_mm expects scale_b to be (1, out_features) and contiguous (rowwise mode).
        self.register_buffer("weight_scale_b", scale_row.reshape(1, fp8_w.shape[0]).contiguous())
        meta_w = torch.empty(self.weight.shape, device='meta', dtype=self.weight.dtype)
        self._parameters.pop('weight')
        self.weight = meta_w
        self.quantized_weight = True

    @classmethod
    def from_linear(cls, linear: nn.Linear, quantize_weight=True):
        new_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
            dtype=linear.weight.dtype
        )
        new_layer = new_layer.to(linear.weight.device)
        with torch.no_grad():
            new_layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                new_layer.bias.copy_(linear.bias)
        if quantize_weight:
            new_layer.quantize_weight()
        return new_layer

    def forward(self, input):
        if self.quantized_weight:
            W_eff = (self.weight_fp8_T, self.weight_scale_b)
        else:
            W_eff = self.weight
        return FP8LinearFunction.apply(
            input, W_eff, self.bias, self.max_input_val, self.max_weight_val)


def contains_substring(str_list, target_str):
    """
    检测 str_list 中是否存在某个字符串被包含在 target_str 中
    :param str_list: list[str]  要检测的字符串列表
    :param target_str: str      指定的字符串
    :return: bool                存在则返回 True，否则 False
    """
    for s in str_list:
        if s in target_str:
            return True
    return False


def replace_linear_with_scaled_fp8(
    module: nn.Module,
    ignore_keys=None,
    quantize_weight: bool = True,
    only_rectangular: bool = False,
    rectangular_mode: str = "any",
):
    """
    Replace nn.Linear layers with FP8ScaleLinear in-place.

    Args:
        ignore_keys: list[str] of module name substrings to skip.
        quantize_weight: quantize weights once at init.
        only_rectangular: if True, skip square linears (in_features == out_features).
            This is a pragmatic stability knob for transformer blocks where FP8 on
            attention projections can sometimes collapse outputs.
        rectangular_mode: optional extra filter for rectangular linears:
            "any" keeps all non-square layers, "expansion" keeps only
            out_features > in_features, and "contraction" keeps only
            out_features < in_features.
    """
    if ignore_keys is None:
        ignore_keys = []
    rectangular_mode = str(rectangular_mode or "any").strip().lower()
    if rectangular_mode in {"expand", "expanding", "up", "out_gt_in"}:
        rectangular_mode = "expansion"
    elif rectangular_mode in {"contract", "contracting", "down", "out_lt_in"}:
        rectangular_mode = "contraction"
    elif rectangular_mode not in {"any", "expansion", "contraction"}:
        rectangular_mode = "any"

    if len(ignore_keys) > 0:
        for name, child in module.named_modules():
            if isinstance(child, nn.Linear) and contains_substring(ignore_keys, name):
                setattr(child, "need_fp8", False)

    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not hasattr(child, "need_fp8"):
            if only_rectangular and child.in_features == child.out_features:
                continue
            if only_rectangular and rectangular_mode == "expansion" and child.out_features <= child.in_features:
                continue
            if only_rectangular and rectangular_mode == "contraction" and child.out_features >= child.in_features:
                continue
            setattr(module, name, FP8ScaleLinear.from_linear(child, quantize_weight=quantize_weight))
        else:
            replace_linear_with_scaled_fp8(
                child,
                ignore_keys=ignore_keys,
                quantize_weight=quantize_weight,
                only_rectangular=only_rectangular,
                rectangular_mode=rectangular_mode,
            )
