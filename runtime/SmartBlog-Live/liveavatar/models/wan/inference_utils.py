import logging
import os
import inspect
from functools import wraps

import torch

STREAMING_VAE = True
COMPILE = os.getenv("ENABLE_COMPILE", "true").lower() == "true"
print(f"COMPILE: {COMPILE}")
torch._dynamo.config.cache_size_limit = int(os.getenv("TORCHDYNAMO_CACHE_SIZE_LIMIT", "128") or 128)
if hasattr(torch._dynamo.config, "capture_scalar_outputs"):
    _cap_scalars = str(os.getenv("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", "0") or "0").strip().lower()
    torch._dynamo.config.capture_scalar_outputs = _cap_scalars in ("1", "true", "yes", "on")

# Avoid massive warning spam when torch.compile falls back internally on unsupported dynamic patterns.
try:
    if hasattr(torch, "_logging") and hasattr(torch._logging, "set_logs"):
        torch._logging.set_logs(dynamo=logging.ERROR)
except Exception:
    pass

NO_REFRESH_INFERENCE = False

def is_compile_supported():
    return hasattr(torch, "compiler") and hasattr(torch.nn.Module, "compile")

def disable(func):
    if is_compile_supported():
        return torch.compiler.disable(func)
    return func

def _compile_dynamic_arg() -> bool | None:
    raw = str(os.getenv("TORCH_COMPILE_DYNAMIC", "1") or "1").strip().lower()
    if raw in ("", "none", "auto", "default"):
        return None
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def _compile_skip_tokens() -> list[str]:
    raw = str(os.getenv("TORCH_COMPILE_SKIP_FUNCS", "") or "").strip()
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _compile_include_tokens() -> list[str]:
    raw = str(os.getenv("TORCH_COMPILE_INCLUDE_FUNCS", "") or "").strip()
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return bool(default)


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _configure_inductor_for_compile() -> None:
    if not COMPILE:
        return
    try:
        cfg = torch._inductor.config
    except Exception:
        return

    applied: list[str] = []

    def _set_attr(obj, attr: str, value, label: str) -> None:
        if hasattr(obj, attr):
            try:
                setattr(obj, attr, value)
                applied.append(f"{label}={value}")
            except Exception:
                pass

    _set_attr(cfg, "fx_graph_cache", _env_bool("TORCHINDUCTOR_FX_GRAPH_CACHE", True), "fx_graph_cache")
    _set_attr(cfg, "max_autotune", _env_bool("TORCHINDUCTOR_MAX_AUTOTUNE", True), "max_autotune")
    _set_attr(
        cfg,
        "max_autotune_pointwise",
        _env_bool("TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE", True),
        "max_autotune_pointwise",
    )
    _set_attr(cfg, "max_autotune_gemm", _env_bool("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", True), "max_autotune_gemm")
    _set_attr(
        cfg,
        "coordinate_descent_tuning",
        _env_bool("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING", True),
        "coordinate_descent_tuning",
    )
    _set_attr(cfg, "compile_threads", _env_int("TORCHINDUCTOR_COMPILE_THREADS", 32), "compile_threads")

    triton_cfg = getattr(cfg, "triton", None)
    if triton_cfg is not None:
        _set_attr(
            triton_cfg,
            "cudagraphs",
            _env_bool("TORCHINDUCTOR_CUDAGRAPHS", False),
            "triton.cudagraphs",
        )
        _set_attr(
            triton_cfg,
            "cudagraph_trees",
            _env_bool("TORCHINDUCTOR_TRITON_CUDAGRAPH_TREES", False),
            "triton.cudagraph_trees",
        )
        _set_attr(
            triton_cfg,
            "autotune_at_compile_time",
            _env_bool("TORCHINDUCTOR_TRITON_AUTOTUNE_AT_COMPILE_TIME", True),
            "triton.autotune_at_compile_time",
        )
        _set_attr(
            triton_cfg,
            "autotune_with_sample_inputs",
            _env_bool("TORCHINDUCTOR_TRITON_AUTOTUNE_WITH_SAMPLE_INPUTS", False),
            "triton.autotune_with_sample_inputs",
        )
        _set_attr(
            triton_cfg,
            "cudagraph_or_error",
            _env_bool("TORCHINDUCTOR_TRITON_CUDAGRAPH_OR_ERROR", False),
            "triton.cudagraph_or_error",
        )

    if applied:
        logging.info("torch.compile inductor config: %s", ", ".join(applied))


_configure_inductor_for_compile()


def _compile_func_descriptor(func) -> str:
    try:
        src = inspect.getsourcefile(func) or ""
    except Exception:
        src = ""
    qual = str(getattr(func, "__qualname__", getattr(func, "__name__", "unknown")) or "unknown")
    return f"{src}:{qual}"

def conditional_compile(func):
    desc = _compile_func_descriptor(func)
    include_tokens = _compile_include_tokens()
    if include_tokens and not any(tok in desc for tok in include_tokens):
        logging.info("torch.compile skipped for %s (not in include list)", desc)
        return disable(func)
    skip_tokens = _compile_skip_tokens()
    if skip_tokens:
        for tok in skip_tokens:
            if tok in desc:
                logging.info("torch.compile skipped for %s (matched %s)", desc, tok)
                return disable(func)
    if COMPILE:
        mode_raw = str(os.getenv("TORCH_COMPILE_MODE", "") or "").strip()
        backend_raw = str(os.getenv("TORCH_COMPILE_BACKEND", "inductor") or "inductor").strip() or "inductor"
        mode_arg = mode_raw if mode_raw else None
        dynamic_arg = _compile_dynamic_arg()
        try:
            compiled = torch.compile(mode=mode_arg, backend=backend_raw, dynamic=dynamic_arg)(func)
        except Exception as e:
            raise RuntimeError(
                "torch.compile failed "
                f"(backend={backend_raw} mode={mode_arg if mode_arg is not None else 'None'} "
                f"dynamic={dynamic_arg if dynamic_arg is not None else 'None'})"
            ) from e
        if not _env_bool("TORCH_COMPILE_SAFE_FALLBACK", True):
            return compiled

        disabled = False

        @wraps(func)
        def _safe_compiled(*args, **kwargs):
            nonlocal disabled
            if disabled:
                return func(*args, **kwargs)
            try:
                return compiled(*args, **kwargs)
            except Exception:
                disabled = True
                logging.exception(
                    "torch.compile runtime failed for %s; falling back to eager for this function",
                    desc,
                )
                return func(*args, **kwargs)

        return _safe_compiled
    return func
