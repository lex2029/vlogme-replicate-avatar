from __future__ import annotations

import math
import os
import subprocess
import wave

import numpy as np


def to_wav_16k_mono(in_audio_path: str, out_wav_path: str) -> str:
    os.makedirs(os.path.dirname(out_wav_path) or ".", exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            in_audio_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            out_wav_path,
        ],
        check=True,
    )
    return out_wav_path


def wav_duration_seconds(wav_path: str) -> float:
    frames, rate = wav_sample_count(wav_path)
    if rate <= 0:
        return 0.0
    return float(frames) / float(rate)


def wav_sample_count(wav_path: str) -> tuple[int, int]:
    try:
        with wave.open(wav_path, "rb") as wf:
            frames = int(wf.getnframes() or 0)
            rate = int(wf.getframerate() or 0)
        return max(0, frames), max(0, rate)
    except Exception:
        return 0, 0


def wav_audible_sample_count(
    wav_path: str,
    *,
    silence_db: float = -50.0,
    tail_keep_sec: float = 0.12,
) -> int:
    total_samples, rate = wav_sample_count(wav_path)
    if total_samples <= 0 or rate <= 0:
        return 0
    try:
        threshold = int(
            max(1.0, min(32767.0, 32767.0 * (10.0 ** (float(silence_db) / 20.0))))
        )
    except Exception:
        threshold = 104
    try:
        keep_samples = int(max(0.0, float(tail_keep_sec)) * float(rate))
    except Exception:
        keep_samples = int(0.12 * float(rate))
    try:
        with wave.open(wav_path, "rb") as wf:
            channels = max(1, int(wf.getnchannels() or 1))
            sample_width = int(wf.getsampwidth() or 0)
            if sample_width != 2:
                return int(total_samples)
            pcm = wf.readframes(int(total_samples))
        if not pcm:
            return int(total_samples)
        arr = np.frombuffer(pcm, dtype="<i2")
        if arr.size <= 0:
            return int(total_samples)
        if channels > 1:
            usable = int(arr.size // channels) * channels
            if usable <= 0:
                return int(total_samples)
            arr = arr[:usable].reshape((-1, channels))
            levels = np.max(np.abs(arr.astype(np.int32)), axis=1)
        else:
            levels = np.abs(arr.astype(np.int32))
        idx = np.flatnonzero(levels > int(threshold))
        if idx.size <= 0:
            return int(total_samples)
        last_audible = int(idx[-1]) + 1
        return int(max(1, min(int(total_samples), int(last_audible + keep_samples))))
    except Exception:
        return int(total_samples)


def auto_num_clip_for_duration(duration_sec: float, fps: int, infer_frames: int) -> int:
    fps = max(1, int(fps))
    infer_frames = max(1, int(infer_frames))
    dur = float(duration_sec or 0.0)
    if dur <= 0.0:
        raise RuntimeError(f"Invalid or empty audio duration: {duration_sec!r}")

    try:
        pad_sec = float(os.getenv("WORKER_AUDIO_NUM_CLIP_PAD_SEC", "0.25") or 0.25)
    except Exception:
        pad_sec = 0.25
    pad_sec = max(0.0, min(5.0, float(pad_sec)))

    try:
        eff_clip_frames = int(os.getenv("WORKER_AUDIO_EFFECTIVE_CLIP_FRAMES", "0") or 0)
    except Exception:
        eff_clip_frames = 0
    if eff_clip_frames <= 0:
        # One generated clip contributes `infer_frames` frames to the saved
        # offline video. Using fps here overestimates clip count and wastes
        # generation time before ffmpeg trims the output back to the audio.
        eff_clip_frames = int(infer_frames)
    eff_clip_frames = max(1, int(eff_clip_frames))

    frames_needed = int(math.ceil((float(dur) + float(pad_sec)) * float(fps)))
    return max(1, int(math.ceil(float(frames_needed) / float(eff_clip_frames))))


def auto_num_clip_for_audio(audio_wav_path: str, fps: int, infer_frames: int) -> int:
    dur = wav_duration_seconds(audio_wav_path)
    try:
        return auto_num_clip_for_duration(float(dur), fps=fps, infer_frames=infer_frames)
    except RuntimeError:
        raise RuntimeError(f"Invalid or empty WAV duration: {audio_wav_path}")


def ensure_silence_wav(out_wav_path: str, samples: int, sample_rate: int = 16000) -> str:
    if os.path.exists(out_wav_path):
        return out_wav_path
    os.makedirs(os.path.dirname(out_wav_path) or ".", exist_ok=True)
    samples = max(0, int(samples))
    with wave.open(out_wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        if samples > 0:
            wf.writeframes((np.zeros(samples, dtype=np.int16)).tobytes())
        else:
            wf.writeframes(b"")
    return out_wav_path


def normalize_infer_frames(infer_frames: int, *, world_size: int, cfg) -> tuple[int, str | None]:
    try:
        infer_frames = int(infer_frames)
    except Exception:
        infer_frames = 48
    infer_frames = max(4, infer_frames)

    if int(world_size) <= 1:
        return infer_frames, None

    try:
        frames_per_block = int(getattr(cfg, "num_frames_per_block", 3) or 3)
    except Exception:
        frames_per_block = 3
    multiple = max(1, frames_per_block) * 4

    try:
        max_frames = int(os.getenv("INFER_FRAMES_MAX", "160") or 160)
    except Exception:
        max_frames = 160
    max_frames = max(multiple, max_frames)

    try:
        min_frames = int(os.getenv("INFER_FRAMES_MIN_TPP", str(max(24, multiple))) or max(24, multiple))
    except Exception:
        min_frames = max(24, multiple)
    min_frames = max(multiple, min_frames)

    if infer_frames % multiple == 0 and infer_frames >= min_frames:
        return infer_frames, None

    up = int(math.ceil(infer_frames / float(multiple)) * multiple)
    if up > max_frames:
        up = int(math.floor(infer_frames / float(multiple)) * multiple)
    norm = max(min_frames, min(max_frames, up))
    if norm % multiple != 0:
        norm = int(math.floor(norm / float(multiple)) * multiple)
        norm = max(multiple, norm)

    return (
        norm,
        (
            f"TPP requires Frames per Clip to be a multiple of {multiple} "
            f"(num_frames_per_block={frames_per_block}). Adjusted {infer_frames} -> {norm}."
        ),
    )
