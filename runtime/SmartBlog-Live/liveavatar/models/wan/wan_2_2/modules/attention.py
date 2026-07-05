# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import os

import torch


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


_FLASH_ATTN_DISABLED = _env_flag("LIVEAVATAR_DISABLE_FLASH_ATTN", default=False)
_CUDNN_ATTN_DISABLED = _env_flag("LIVEAVATAR_DISABLE_CUDNN_ATTN", default=False)
_TORCH_SDPA_MATH_ONLY = _env_flag("LIVEAVATAR_FORCE_TORCH_SDPA_MATH", default=False)
_EAGER_ATTN_FORCED = _env_flag("LIVEAVATAR_FORCE_EAGER_ATTN", default=False)


def _torch_sdp_backend_enabled(name: str) -> bool | None:
    cuda_backends = getattr(torch.backends, "cuda", None)
    checker = getattr(cuda_backends, f"{name}_sdp_enabled", None)
    if not callable(checker):
        return None
    try:
        return bool(checker())
    except Exception:
        return None


def _set_torch_sdp_backend(name: str, enabled: bool) -> None:
    cuda_backends = getattr(torch.backends, "cuda", None)
    setter = getattr(cuda_backends, f"enable_{name}_sdp", None)
    if not callable(setter):
        return
    try:
        setter(bool(enabled))
    except Exception as e:
        logging.warning("Failed to set torch %s SDPA backend to %s: %s", name, int(bool(enabled)), e)


def _configure_torch_sdp_backends() -> None:
    if _CUDNN_ATTN_DISABLED or _TORCH_SDPA_MATH_ONLY:
        _set_torch_sdp_backend("cudnn", False)
    if not _TORCH_SDPA_MATH_ONLY:
        return
    _set_torch_sdp_backend("flash", False)
    _set_torch_sdp_backend("mem_efficient", False)
    _set_torch_sdp_backend("math", True)


def _torch_cudnn_sdp_enabled() -> bool | None:
    return _torch_sdp_backend_enabled("cudnn")


def _torch_flash_sdp_enabled() -> bool | None:
    return _torch_sdp_backend_enabled("flash")


def _torch_mem_efficient_sdp_enabled() -> bool | None:
    return _torch_sdp_backend_enabled("mem_efficient")


def _torch_math_sdp_enabled() -> bool | None:
    return _torch_sdp_backend_enabled("math")


_configure_torch_sdp_backends()

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

if _FLASH_ATTN_DISABLED:
    FLASH_ATTN_2_AVAILABLE = False
    FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'cudnn_attention_forward_with_lse'
]


_LOGGED_ATTN_BACKENDS = set()


def _log_attention_backend_once(name: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> None:
    if name in _LOGGED_ATTN_BACKENDS:
        return
    _LOGGED_ATTN_BACKENDS.add(name)
    try:
        logging.info(
            "Wan attention backend: backend=%s q=%s k=%s v=%s dtype=%s causal=%s flash2=%s flash3=%s cudnn_sdpa=%s",
            name,
            list(q.shape),
            list(k.shape),
            list(v.shape),
            str(q.dtype),
            int(bool(causal)),
            int(bool(FLASH_ATTN_2_AVAILABLE)),
            int(bool(FLASH_ATTN_3_AVAILABLE)),
            int(bool(_cudnn_sdpa_available())),
        )
    except Exception:
        pass


def _cudnn_sdpa_available() -> bool:
    if _EAGER_ATTN_FORCED:
        return False
    if _CUDNN_ATTN_DISABLED:
        return False
    enabled = _torch_cudnn_sdp_enabled()
    if enabled is False:
        return False
    return bool(hasattr(torch.ops.aten, "_scaled_dot_product_cudnn_attention"))


def _eager_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    softmax_scale = None,
    q_scale = None,
    causal: bool = False,
    dtype=torch.bfloat16,
) -> torch.Tensor:
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    if q_scale is not None:
        q = q * q_scale

    scale = float(softmax_scale) if softmax_scale is not None else q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        lq, lk = int(q.shape[-2]), int(k.shape[-2])
        q_pos = torch.arange(lq, device=q.device) + max(0, lk - lq)
        k_pos = torch.arange(lk, device=q.device)
        causal_mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)

    attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    if dropout_p:
        attn = torch.nn.functional.dropout(attn, p=float(dropout_p), training=True)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2).contiguous()


def cudnn_attention_forward_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale = None,
) -> torch.Tensor:
    # print(f"{q.shape=} {k.shape=}  {v.shape=}")

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = torch.ops.aten._scaled_dot_product_cudnn_attention(
        q,
        k,
        v,
        attn_bias=attn_mask,
        compute_log_sumexp=True,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    if isinstance(out, tuple):
        out = out[0]
    out = out.transpose(1, 2).contiguous()
    return out


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if _EAGER_ATTN_FORCED:
        _log_attention_backend_once("eager", q, k, v, causal)
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using eager attention. This matches the previous torch_sdpa fallback behavior.'
            )
        return _eager_attention_forward(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )

    def cudnn_require():
        if not _cudnn_sdpa_available():
            return False
        if window_size != (-1, -1):
            return False
        if not q.shape[-1] <= 256:
            return False
        return True

    if cudnn_require():
        _log_attention_backend_once("cudnn_sdpa", q, k, v, causal)
        return cudnn_attention_forward_with_lse(q, k, v, is_causal=causal)
    elif FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        _log_attention_backend_once("flash_attn", q, k, v, causal)
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        _log_attention_backend_once("torch_sdpa", q, k, v, causal)
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
