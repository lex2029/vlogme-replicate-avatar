from __future__ import annotations

import numpy as np


def blend_filler_toward_previous(
    *,
    current: np.ndarray,
    previous: np.ndarray | None,
    similarity: float,
) -> np.ndarray:
    curr = np.asarray(current, dtype=np.float32)
    prev = None if previous is None else np.asarray(previous, dtype=np.float32)
    if prev is None or prev.ndim != 1 or curr.ndim != 1 or prev.size != curr.size:
        return np.ascontiguousarray(curr, dtype=np.float32)
    sim = float(max(0.0, min(0.9995, similarity)))
    if sim <= 0.0:
        return np.ascontiguousarray(curr, dtype=np.float32)
    out = (prev * sim) + (curr * (1.0 - sim))
    target_rms = float(np.sqrt(np.mean(np.square(curr), dtype=np.float64)))
    out_rms = float(np.sqrt(np.mean(np.square(out), dtype=np.float64)))
    if target_rms > 1e-12 and out_rms > 1e-12:
        out *= float(target_rms / out_rms)
    np.clip(out, -1.0, 1.0, out=out)
    return np.ascontiguousarray(out, dtype=np.float32)


def build_filler_pcm_f32(
    *,
    samples: int,
    mode: str,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    samples_n = max(1, int(samples))
    mode_norm = str(mode or "silence").strip().lower()
    if mode_norm in {"noise", "smooth_noise"} and float(noise_std) > 0.0:
        rng = np.random.default_rng(int(seed))
        arr = rng.standard_normal(int(samples_n)).astype(np.float32)
        if mode_norm == "smooth_noise" and samples_n > 2:
            # White noise makes idle motion jittery. For long-idle keepalive we
            # want a very low-energy, slowly changing signal so the model keeps
            # moving a little without "emoting" on every chunk. Keep it
            # noticeably smoother than hiss so the spectral energy does not jump
            # around from chunk to chunk.
            win = min(1024, max(32, samples_n // 16))
            kernel = np.ones((int(win),), dtype=np.float32) / float(win)
            arr = np.convolve(arr, kernel, mode="same").astype(np.float32, copy=False)
            rms = float(np.sqrt(np.mean(np.square(arr), dtype=np.float64)))
            if rms > 1e-12:
                arr *= float(1.0 / rms)
        arr *= float(noise_std)
        np.clip(arr, -1.0, 1.0, out=arr)
        return np.ascontiguousarray(arr, dtype=np.float32)
    return np.zeros((samples_n,), dtype=np.float32)
