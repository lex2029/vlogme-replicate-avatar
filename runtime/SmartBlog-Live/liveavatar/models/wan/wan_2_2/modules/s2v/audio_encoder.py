# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import logging
import os

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


_AUDIO_PROFILE_LOGGED = False


def get_sample_indices(original_fps,
                       total_frames,
                       target_fps,
                       num_sample,
                       fixed_start=None):
    required_duration = num_sample / target_fps
    required_origin_frames = int(np.ceil(required_duration * original_fps))
    if required_duration > total_frames / original_fps:
        raise ValueError("required_duration must be less than video length")

    if not fixed_start is None and fixed_start >= 0:
        start_frame = fixed_start
    else:
        max_start = total_frames - required_origin_frames
        if max_start < 0:
            raise ValueError("video length is too short")
        start_frame = np.random.randint(0, max_start + 1)
    start_time = start_frame / original_fps

    end_time = start_time + required_duration
    time_points = np.linspace(start_time, end_time, num_sample, endpoint=False)

    frame_indices = np.round(np.array(time_points) * original_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    return frame_indices


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """
    features: shape=[1, T, 512]
    input_fps: fps for audio, f_a
    output_fps: fps for video, f_m
    output_len: video length
    """
    features = features.transpose(1, 2)  # [1, 512, T]
    seq_len = features.shape[2] / float(input_fps)  # T/f_a
    if output_len is None:
        output_len = int(seq_len * output_fps)  # f_m*T/f_a
    output_features = F.interpolate(
        features, size=output_len, align_corners=True,
        mode='linear')  # [1, 512, output_len]
    return output_features.transpose(1, 2)  # [1, output_len, 512]


_AUDIO_PROFILE_DEFAULTS = {
    "off": {
        "LIVEAVATAR_MODEL_AUDIO_PREEMPHASIS": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_TARGET_RMS": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_MAX_GAIN": 2.0,
        "LIVEAVATAR_MODEL_AUDIO_GAIN": 1.0,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP": 0,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP_DRIVE": 1.0,
        "LIVEAVATAR_MODEL_AUDIO_ORIGINAL_MIX": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_ADVANCE_MS": 0.0,
    },
    "baseline": {
        "LIVEAVATAR_MODEL_AUDIO_PREEMPHASIS": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_TARGET_RMS": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_MAX_GAIN": 2.0,
        "LIVEAVATAR_MODEL_AUDIO_GAIN": 1.0,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP": 0,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP_DRIVE": 1.0,
        "LIVEAVATAR_MODEL_AUDIO_ORIGINAL_MIX": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_ADVANCE_MS": 0.0,
    },
    "clear": {
        "LIVEAVATAR_MODEL_AUDIO_PREEMPHASIS": 0.35,
        "LIVEAVATAR_MODEL_AUDIO_TARGET_RMS": 0.095,
        "LIVEAVATAR_MODEL_AUDIO_MAX_GAIN": 2.5,
        "LIVEAVATAR_MODEL_AUDIO_GAIN": 1.0,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP": 1,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP_DRIVE": 1.25,
        "LIVEAVATAR_MODEL_AUDIO_ORIGINAL_MIX": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_ADVANCE_MS": 0.0,
    },
    "punchy": {
        "LIVEAVATAR_MODEL_AUDIO_PREEMPHASIS": 0.65,
        "LIVEAVATAR_MODEL_AUDIO_TARGET_RMS": 0.12,
        "LIVEAVATAR_MODEL_AUDIO_MAX_GAIN": 3.0,
        "LIVEAVATAR_MODEL_AUDIO_GAIN": 1.10,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP": 1,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP_DRIVE": 1.8,
        "LIVEAVATAR_MODEL_AUDIO_ORIGINAL_MIX": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_ADVANCE_MS": 55.0,
    },
    "natural": {
        "LIVEAVATAR_MODEL_AUDIO_PREEMPHASIS": 0.72,
        "LIVEAVATAR_MODEL_AUDIO_TARGET_RMS": 0.13,
        "LIVEAVATAR_MODEL_AUDIO_MAX_GAIN": 3.2,
        "LIVEAVATAR_MODEL_AUDIO_GAIN": 1.10,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP": 1,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP_DRIVE": 1.75,
        "LIVEAVATAR_MODEL_AUDIO_ORIGINAL_MIX": 0.35,
        "LIVEAVATAR_MODEL_AUDIO_ADVANCE_MS": 70.0,
    },
    "quickmouth": {
        "LIVEAVATAR_MODEL_AUDIO_PREEMPHASIS": 0.70,
        "LIVEAVATAR_MODEL_AUDIO_TARGET_RMS": 0.13,
        "LIVEAVATAR_MODEL_AUDIO_MAX_GAIN": 3.2,
        "LIVEAVATAR_MODEL_AUDIO_GAIN": 1.10,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_BOOST": 0.45,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_FAST_MS": 18.0,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_SLOW_MS": 120.0,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_EXPAND": 0.85,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_THRESHOLD": 0.040,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_FLOOR": 0.18,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP": 1,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP_DRIVE": 1.65,
        "LIVEAVATAR_MODEL_AUDIO_ORIGINAL_MIX": 0.30,
        "LIVEAVATAR_MODEL_AUDIO_ADVANCE_MS": 70.0,
    },
    "quickmouth_plus": {
        "LIVEAVATAR_MODEL_AUDIO_PREEMPHASIS": 0.72,
        "LIVEAVATAR_MODEL_AUDIO_TARGET_RMS": 0.135,
        "LIVEAVATAR_MODEL_AUDIO_MAX_GAIN": 3.4,
        "LIVEAVATAR_MODEL_AUDIO_GAIN": 1.12,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_BOOST": 0.72,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_FAST_MS": 14.0,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_SLOW_MS": 135.0,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_EXPAND": 1.25,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_THRESHOLD": 0.050,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_FLOOR": 0.10,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP": 1,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP_DRIVE": 1.70,
        "LIVEAVATAR_MODEL_AUDIO_ORIGINAL_MIX": 0.22,
        "LIVEAVATAR_MODEL_AUDIO_ADVANCE_MS": 75.0,
    },
    "quickmouth_hard": {
        "LIVEAVATAR_MODEL_AUDIO_PREEMPHASIS": 0.74,
        "LIVEAVATAR_MODEL_AUDIO_TARGET_RMS": 0.14,
        "LIVEAVATAR_MODEL_AUDIO_MAX_GAIN": 3.6,
        "LIVEAVATAR_MODEL_AUDIO_GAIN": 1.14,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_BOOST": 1.00,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_FAST_MS": 10.0,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_SLOW_MS": 150.0,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_EXPAND": 1.70,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_THRESHOLD": 0.060,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_FLOOR": 0.06,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP": 1,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP_DRIVE": 1.78,
        "LIVEAVATAR_MODEL_AUDIO_ORIGINAL_MIX": 0.15,
        "LIVEAVATAR_MODEL_AUDIO_ADVANCE_MS": 80.0,
    },
    "quickmouth_presence": {
        "LIVEAVATAR_MODEL_AUDIO_PREEMPHASIS": 0.76,
        "LIVEAVATAR_MODEL_AUDIO_TARGET_RMS": 0.14,
        "LIVEAVATAR_MODEL_AUDIO_MAX_GAIN": 3.6,
        "LIVEAVATAR_MODEL_AUDIO_GAIN": 1.14,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_BOOST": 1.05,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_FAST_MS": 9.0,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_SLOW_MS": 150.0,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_EXPAND": 1.85,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_THRESHOLD": 0.065,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_FLOOR": 0.05,
        "LIVEAVATAR_MODEL_AUDIO_PRESENCE_BOOST": 0.42,
        "LIVEAVATAR_MODEL_AUDIO_PRESENCE_LOW_HZ": 650.0,
        "LIVEAVATAR_MODEL_AUDIO_PRESENCE_HIGH_HZ": 3600.0,
        "LIVEAVATAR_MODEL_AUDIO_PRESENCE_EDGE_HZ": 250.0,
        "LIVEAVATAR_MODEL_AUDIO_PRESENCE_SOFT_THRESHOLD": 0.13,
        "LIVEAVATAR_MODEL_AUDIO_PRESENCE_LOUD_THRESHOLD": 0.27,
        "LIVEAVATAR_MODEL_AUDIO_PRESENCE_CURVE": 1.10,
        "LIVEAVATAR_MODEL_AUDIO_CONSONANT_BOOST": 0.22,
        "LIVEAVATAR_MODEL_AUDIO_CONSONANT_LOW_HZ": 2400.0,
        "LIVEAVATAR_MODEL_AUDIO_CONSONANT_HIGH_HZ": 6200.0,
        "LIVEAVATAR_MODEL_AUDIO_CONSONANT_EDGE_HZ": 400.0,
        "LIVEAVATAR_MODEL_AUDIO_CONSONANT_SOFT_THRESHOLD": 0.10,
        "LIVEAVATAR_MODEL_AUDIO_CONSONANT_LOUD_THRESHOLD": 0.22,
        "LIVEAVATAR_MODEL_AUDIO_CONSONANT_CURVE": 1.25,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP": 1,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP_DRIVE": 1.82,
        "LIVEAVATAR_MODEL_AUDIO_ORIGINAL_MIX": 0.12,
        "LIVEAVATAR_MODEL_AUDIO_ADVANCE_MS": 82.0,
    },
    "quickmouth_allsharp": {
        "LIVEAVATAR_MODEL_AUDIO_PREEMPHASIS": 0.78,
        "LIVEAVATAR_MODEL_AUDIO_TARGET_RMS": 0.145,
        "LIVEAVATAR_MODEL_AUDIO_MAX_GAIN": 3.8,
        "LIVEAVATAR_MODEL_AUDIO_GAIN": 1.16,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_BOOST": 1.10,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_FAST_MS": 8.0,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_SLOW_MS": 160.0,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_EXPAND": 2.00,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_THRESHOLD": 0.070,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_FLOOR": 0.045,
        "LIVEAVATAR_MODEL_AUDIO_VOWEL_BOOST": 0.30,
        "LIVEAVATAR_MODEL_AUDIO_VOWEL_LOW_HZ": 180.0,
        "LIVEAVATAR_MODEL_AUDIO_VOWEL_HIGH_HZ": 950.0,
        "LIVEAVATAR_MODEL_AUDIO_VOWEL_EDGE_HZ": 120.0,
        "LIVEAVATAR_MODEL_AUDIO_VOWEL_SOFT_THRESHOLD": 0.16,
        "LIVEAVATAR_MODEL_AUDIO_VOWEL_LOUD_THRESHOLD": 0.33,
        "LIVEAVATAR_MODEL_AUDIO_VOWEL_CURVE": 1.00,
        "LIVEAVATAR_MODEL_AUDIO_FORMANT_BOOST": 0.36,
        "LIVEAVATAR_MODEL_AUDIO_FORMANT_LOW_HZ": 850.0,
        "LIVEAVATAR_MODEL_AUDIO_FORMANT_HIGH_HZ": 2300.0,
        "LIVEAVATAR_MODEL_AUDIO_FORMANT_EDGE_HZ": 220.0,
        "LIVEAVATAR_MODEL_AUDIO_FORMANT_SOFT_THRESHOLD": 0.14,
        "LIVEAVATAR_MODEL_AUDIO_FORMANT_LOUD_THRESHOLD": 0.29,
        "LIVEAVATAR_MODEL_AUDIO_FORMANT_CURVE": 1.05,
        "LIVEAVATAR_MODEL_AUDIO_PRESENCE_BOOST": 0.46,
        "LIVEAVATAR_MODEL_AUDIO_PRESENCE_LOW_HZ": 650.0,
        "LIVEAVATAR_MODEL_AUDIO_PRESENCE_HIGH_HZ": 3800.0,
        "LIVEAVATAR_MODEL_AUDIO_PRESENCE_EDGE_HZ": 250.0,
        "LIVEAVATAR_MODEL_AUDIO_PRESENCE_SOFT_THRESHOLD": 0.13,
        "LIVEAVATAR_MODEL_AUDIO_PRESENCE_LOUD_THRESHOLD": 0.27,
        "LIVEAVATAR_MODEL_AUDIO_PRESENCE_CURVE": 1.15,
        "LIVEAVATAR_MODEL_AUDIO_CONSONANT_BOOST": 0.26,
        "LIVEAVATAR_MODEL_AUDIO_CONSONANT_LOW_HZ": 2400.0,
        "LIVEAVATAR_MODEL_AUDIO_CONSONANT_HIGH_HZ": 6400.0,
        "LIVEAVATAR_MODEL_AUDIO_CONSONANT_EDGE_HZ": 400.0,
        "LIVEAVATAR_MODEL_AUDIO_CONSONANT_SOFT_THRESHOLD": 0.10,
        "LIVEAVATAR_MODEL_AUDIO_CONSONANT_LOUD_THRESHOLD": 0.22,
        "LIVEAVATAR_MODEL_AUDIO_CONSONANT_CURVE": 1.30,
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_PUFF": 0.20,
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_CLOSE": 0.58,
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_RELEASE": 0.24,
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_PRE_CLOSE_MS": 26.0,
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_PUFF_MS": 30.0,
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_COOLDOWN_MS": 90.0,
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_SCORE_THRESHOLD": 0.20,
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_ACTIVITY_FLOOR": 0.018,
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_LOW_HZ": 90.0,
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_HIGH_HZ": 950.0,
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_EDGE_HZ": 160.0,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP": 1,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP_DRIVE": 1.88,
        "LIVEAVATAR_MODEL_AUDIO_ORIGINAL_MIX": 0.08,
        "LIVEAVATAR_MODEL_AUDIO_ADVANCE_MS": 84.0,
    },
    "avatar_clarity": {
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_CHAIN": 1,
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_AMOUNT": 1.35,
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_HIGHPASS_HZ": 105.0,
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_MUD_CUT": 0.30,
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_VOWEL_BOOST": 0.24,
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_INTELLIGIBILITY_BOOST": 0.44,
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_PRESENCE_BOOST": 0.32,
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_DEESS": 0.10,
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_COMPRESSOR": 1,
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_COMPRESS_THRESHOLD": 0.08,
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_COMPRESS_RATIO": 4.0,
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_TARGET_RMS": 0.17,
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_MAX_GAIN": 4.0,
        "LIVEAVATAR_MODEL_AUDIO_PREEMPHASIS": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_TARGET_RMS": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_MAX_GAIN": 2.0,
        "LIVEAVATAR_MODEL_AUDIO_GAIN": 1.0,
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_BOOST": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_QUIET_EXPAND": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_VOWEL_BOOST": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_FORMANT_BOOST": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_PRESENCE_BOOST": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_CONSONANT_BOOST": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_PUFF": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP": 1,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP_DRIVE": 1.65,
        "LIVEAVATAR_MODEL_AUDIO_ORIGINAL_MIX": 0.02,
        "LIVEAVATAR_MODEL_AUDIO_ADVANCE_MS": 0.0,
    },
    "sharp": {
        "LIVEAVATAR_MODEL_AUDIO_PREEMPHASIS": 0.82,
        "LIVEAVATAR_MODEL_AUDIO_TARGET_RMS": 0.14,
        "LIVEAVATAR_MODEL_AUDIO_MAX_GAIN": 3.5,
        "LIVEAVATAR_MODEL_AUDIO_GAIN": 1.15,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP": 1,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP_DRIVE": 2.35,
        "LIVEAVATAR_MODEL_AUDIO_ORIGINAL_MIX": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_ADVANCE_MS": 85.0,
    },
    "snappy": {
        "LIVEAVATAR_MODEL_AUDIO_PREEMPHASIS": 0.90,
        "LIVEAVATAR_MODEL_AUDIO_TARGET_RMS": 0.16,
        "LIVEAVATAR_MODEL_AUDIO_MAX_GAIN": 4.0,
        "LIVEAVATAR_MODEL_AUDIO_GAIN": 1.20,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP": 1,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP_DRIVE": 3.0,
        "LIVEAVATAR_MODEL_AUDIO_ORIGINAL_MIX": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_ADVANCE_MS": 115.0,
    },
    "overdrive": {
        "LIVEAVATAR_MODEL_AUDIO_PREEMPHASIS": 0.96,
        "LIVEAVATAR_MODEL_AUDIO_TARGET_RMS": 0.19,
        "LIVEAVATAR_MODEL_AUDIO_MAX_GAIN": 5.0,
        "LIVEAVATAR_MODEL_AUDIO_GAIN": 1.30,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP": 1,
        "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP_DRIVE": 4.0,
        "LIVEAVATAR_MODEL_AUDIO_ORIGINAL_MIX": 0.0,
        "LIVEAVATAR_MODEL_AUDIO_ADVANCE_MS": 145.0,
    },
}


def _audio_profile_defaults():
    profile = str(os.getenv("LIVEAVATAR_MODEL_AUDIO_PROFILE", "custom") or "custom").strip().lower()
    return dict(_AUDIO_PROFILE_DEFAULTS.get(profile, {}))


def _env_float(name, default, low=None, high=None, *, profile_defaults=None):
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        if profile_defaults is None:
            profile_defaults = _audio_profile_defaults()
        value = float(profile_defaults.get(name, default))
    else:
        try:
            value = float(raw)
        except Exception:
            value = float(default)
    if low is not None:
        value = max(float(low), value)
    if high is not None:
        value = min(float(high), value)
    return float(value)


def _env_flag(name, default="0", *, profile_defaults=None):
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        if profile_defaults is None:
            profile_defaults = _audio_profile_defaults()
        raw = str(profile_defaults.get(name, default) or default)
    raw = str(raw).strip().lower()
    return raw in ("1", "true", "yes", "on")


def _smooth_abs_envelope(x, sample_rate, window_ms):
    window = int(round(float(sample_rate) * float(window_ms) / 1000.0))
    if window <= 1:
        return np.abs(x).astype(np.float32, copy=False)
    window = min(int(window), int(max(1, x.size)))
    pad_left = int(window // 2)
    pad_right = int(window - 1 - pad_left)
    padded = np.pad(np.abs(x).astype(np.float32, copy=False), (pad_left, pad_right), mode="edge")
    csum = np.cumsum(np.concatenate(([0.0], padded.astype(np.float64))))
    env = (csum[window:] - csum[:-window]) / float(window)
    return env.astype(np.float32, copy=False)


def _spectral_band(x, sample_rate, low_hz, high_hz, edge_hz):
    if sample_rate <= 0 or x.size < 16:
        return np.zeros_like(x, dtype=np.float32)

    nyquist = float(sample_rate) * 0.5
    low = max(20.0, min(float(low_hz), nyquist - 1.0))
    high = max(low + 1.0, min(float(high_hz), nyquist - 1.0))
    edge = max(0.0, min(float(edge_hz), (high - low) * 0.5))

    spectrum = np.fft.rfft(x.astype(np.float32, copy=False))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate))
    if edge <= 1e-6:
        weights = ((freqs >= low) & (freqs <= high)).astype(np.float32)
    else:
        low_t = np.clip((freqs - low) / edge, 0.0, 1.0)
        high_t = np.clip((high - freqs) / edge, 0.0, 1.0)
        highpass = 0.5 - 0.5 * np.cos(np.pi * low_t)
        lowpass = 0.5 - 0.5 * np.cos(np.pi * high_t)
        weights = (highpass * lowpass).astype(np.float32, copy=False)
        weights[(freqs < low) | (freqs > high)] = 0.0

    return np.fft.irfft(spectrum * weights, n=x.size).astype(np.float32, copy=False)


def _limited_band_boost(
    x,
    sample_rate,
    profile_defaults,
    *,
    prefix,
    low_default,
    high_default,
    soft_default,
    loud_default,
):
    boost = _env_float(
        f"LIVEAVATAR_MODEL_AUDIO_{prefix}_BOOST",
        0.0,
        low=0.0,
        high=2.0,
        profile_defaults=profile_defaults,
    )
    if boost <= 1e-6 or x.size < 16 or sample_rate <= 0:
        return x

    low_hz = _env_float(
        f"LIVEAVATAR_MODEL_AUDIO_{prefix}_LOW_HZ",
        low_default,
        low=20.0,
        high=float(sample_rate) * 0.5 - 2.0,
        profile_defaults=profile_defaults,
    )
    high_hz = _env_float(
        f"LIVEAVATAR_MODEL_AUDIO_{prefix}_HIGH_HZ",
        high_default,
        low=float(low_hz) + 1.0,
        high=float(sample_rate) * 0.5 - 1.0,
        profile_defaults=profile_defaults,
    )
    edge_hz = _env_float(
        f"LIVEAVATAR_MODEL_AUDIO_{prefix}_EDGE_HZ",
        250.0,
        low=0.0,
        high=2000.0,
        profile_defaults=profile_defaults,
    )
    soft_threshold = _env_float(
        f"LIVEAVATAR_MODEL_AUDIO_{prefix}_SOFT_THRESHOLD",
        soft_default,
        low=0.0,
        high=1.0,
        profile_defaults=profile_defaults,
    )
    loud_threshold = _env_float(
        f"LIVEAVATAR_MODEL_AUDIO_{prefix}_LOUD_THRESHOLD",
        loud_default,
        low=float(soft_threshold) + 1e-4,
        high=1.0,
        profile_defaults=profile_defaults,
    )
    curve = _env_float(
        f"LIVEAVATAR_MODEL_AUDIO_{prefix}_CURVE",
        1.0,
        low=0.2,
        high=4.0,
        profile_defaults=profile_defaults,
    )

    band = _spectral_band(x, sample_rate, low_hz, high_hz, edge_hz)
    env = _smooth_abs_envelope(x, sample_rate, 25.0)
    room = max(1e-6, float(loud_threshold) - float(soft_threshold))
    softness = np.clip((float(loud_threshold) - env) / room, 0.0, 1.0)
    if abs(float(curve) - 1.0) > 1e-6:
        softness = np.power(softness, float(curve))

    return x + band * (float(boost) * softness.astype(np.float32, copy=False))


def _apply_avatar_clarity_chain(x, sample_rate, profile_defaults):
    """Clean model-only speech preprocessing for LiveAvatar lip conditioning.

    The final playback audio is not touched. This chain avoids artificial mouth
    cues and only makes the voice drier, flatter, and easier for Wav2Vec to read.
    """
    if not _env_flag("LIVEAVATAR_MODEL_AUDIO_CLARITY_CHAIN", "0", profile_defaults=profile_defaults):
        return x
    if sample_rate <= 0 or x.size < 64:
        return x

    amount = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_AMOUNT",
        1.0,
        low=0.0,
        high=2.0,
        profile_defaults=profile_defaults,
    )
    if amount <= 1e-6:
        return x

    out = x.astype(np.float32, copy=True)

    highpass_hz = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_HIGHPASS_HZ",
        90.0,
        low=0.0,
        high=float(sample_rate) * 0.5 - 2.0,
        profile_defaults=profile_defaults,
    )
    if highpass_hz > 20.0:
        low_band = _spectral_band(out, sample_rate, 20.0, highpass_hz, min(45.0, highpass_hz * 0.5))
        out = out - low_band

    mud_cut = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_MUD_CUT",
        0.0,
        low=0.0,
        high=0.8,
        profile_defaults=profile_defaults,
    )
    if mud_cut > 1e-6:
        out = out - _spectral_band(out, sample_rate, 170.0, 360.0, 90.0) * (float(mud_cut) * float(amount))

    vowel_boost = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_VOWEL_BOOST",
        0.0,
        low=0.0,
        high=1.0,
        profile_defaults=profile_defaults,
    )
    if vowel_boost > 1e-6:
        out = out + _spectral_band(out, sample_rate, 650.0, 1450.0, 220.0) * (float(vowel_boost) * float(amount))

    intelligibility_boost = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_INTELLIGIBILITY_BOOST",
        0.0,
        low=0.0,
        high=1.0,
        profile_defaults=profile_defaults,
    )
    if intelligibility_boost > 1e-6:
        out = out + _spectral_band(out, sample_rate, 2400.0, 4300.0, 360.0) * (
            float(intelligibility_boost) * float(amount)
        )

    presence_boost = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_PRESENCE_BOOST",
        0.0,
        low=0.0,
        high=1.0,
        profile_defaults=profile_defaults,
    )
    if presence_boost > 1e-6:
        out = out + _spectral_band(out, sample_rate, 900.0, 2600.0, 260.0) * (
            float(presence_boost) * float(amount)
        )

    deess = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_DEESS",
        0.0,
        low=0.0,
        high=1.0,
        profile_defaults=profile_defaults,
    )
    if deess > 1e-6:
        sib = _spectral_band(out, sample_rate, 5500.0, min(8500.0, float(sample_rate) * 0.5 - 2.0), 500.0)
        sib_env = _smooth_abs_envelope(sib, sample_rate, 8.0)
        voice_env = _smooth_abs_envelope(out, sample_rate, 28.0)
        excess = np.clip((sib_env - voice_env * 0.45) / (voice_env * 0.80 + 0.025), 0.0, 1.0)
        out = out - sib * (float(deess) * float(amount)) * excess.astype(np.float32, copy=False)

    if _env_flag("LIVEAVATAR_MODEL_AUDIO_CLARITY_COMPRESSOR", "0", profile_defaults=profile_defaults):
        threshold = _env_float(
            "LIVEAVATAR_MODEL_AUDIO_CLARITY_COMPRESS_THRESHOLD",
            0.11,
            low=0.01,
            high=0.8,
            profile_defaults=profile_defaults,
        )
        ratio = _env_float(
            "LIVEAVATAR_MODEL_AUDIO_CLARITY_COMPRESS_RATIO",
            3.0,
            low=1.0,
            high=12.0,
            profile_defaults=profile_defaults,
        )
        env = _smooth_abs_envelope(out, sample_rate, 12.0)
        gain = np.ones_like(out, dtype=np.float32)
        active = env > float(threshold)
        if int(np.count_nonzero(active)) > 0:
            compressed = float(threshold) + (env[active] - float(threshold)) / float(ratio)
            gain[active] = compressed / (env[active] + 1e-6)
            out = out * gain

    target_rms = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_CLARITY_TARGET_RMS",
        0.0,
        low=0.0,
        high=0.5,
        profile_defaults=profile_defaults,
    )
    if target_rms > 1e-6:
        max_gain = _env_float(
            "LIVEAVATAR_MODEL_AUDIO_CLARITY_MAX_GAIN",
            3.0,
            low=0.1,
            high=8.0,
            profile_defaults=profile_defaults,
        )
        rms = float(np.sqrt(np.mean(np.square(out, dtype=np.float32))) + 1e-8)
        if rms > 1e-8:
            out = out * min(float(max_gain), float(target_rms) / rms)

    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


def _apply_bilabial_puff(x, sample_rate, profile_defaults):
    """Add model-only closure/release cues for P/B/M-like consonant onsets."""
    puff = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_PUFF",
        0.0,
        low=0.0,
        high=1.0,
        profile_defaults=profile_defaults,
    )
    if puff <= 1e-6 or sample_rate <= 0 or x.size < 64:
        return x

    close_strength = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_CLOSE",
        0.0,
        low=0.0,
        high=0.9,
        profile_defaults=profile_defaults,
    )
    release_strength = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_RELEASE",
        float(puff),
        low=0.0,
        high=1.0,
        profile_defaults=profile_defaults,
    )
    pre_close_ms = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_PRE_CLOSE_MS",
        24.0,
        low=2.0,
        high=80.0,
        profile_defaults=profile_defaults,
    )
    puff_ms = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_PUFF_MS",
        28.0,
        low=4.0,
        high=90.0,
        profile_defaults=profile_defaults,
    )
    cooldown_ms = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_COOLDOWN_MS",
        90.0,
        low=20.0,
        high=300.0,
        profile_defaults=profile_defaults,
    )
    threshold = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_SCORE_THRESHOLD",
        0.24,
        low=0.0,
        high=4.0,
        profile_defaults=profile_defaults,
    )
    activity_floor = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_ACTIVITY_FLOOR",
        0.018,
        low=0.0,
        high=0.25,
        profile_defaults=profile_defaults,
    )
    max_events_per_sec = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_MAX_EVENTS_PER_SEC",
        9.0,
        low=1.0,
        high=30.0,
        profile_defaults=profile_defaults,
    )

    fast_env = _smooth_abs_envelope(x, sample_rate, 5.0)
    slow_env = _smooth_abs_envelope(x, sample_rate, 85.0)
    speech_ref = float(np.percentile(fast_env, 90.0)) if fast_env.size else 0.0
    if speech_ref <= 1e-6:
        return x

    activity = fast_env / float(speech_ref + 1e-6)
    transient = np.maximum(0.0, fast_env - slow_env) / (slow_env + 0.004)
    score = transient * np.clip(activity, 0.0, 1.8)
    active = fast_env >= float(activity_floor)
    if int(np.count_nonzero(active)) <= 0:
        return x

    active_scores = score[active]
    dynamic_threshold = float(np.percentile(active_scores, 88.0)) if active_scores.size else 0.0
    cutoff = max(float(threshold), float(dynamic_threshold) * 0.70)
    if cutoff <= 1e-6:
        return x

    local_max = np.zeros_like(score, dtype=bool)
    if score.size > 2:
        local_max[1:-1] = (score[1:-1] >= score[:-2]) & (score[1:-1] > score[2:])
    candidates = np.flatnonzero(local_max & active & (score >= cutoff))
    if candidates.size <= 0:
        return x

    cooldown = max(1, int(round(float(sample_rate) * float(cooldown_ms) / 1000.0)))
    selected = []
    last = -cooldown
    for idx in candidates:
        idx_i = int(idx)
        if idx_i - last < cooldown:
            if selected and float(score[idx_i]) > float(score[selected[-1]]):
                selected[-1] = idx_i
                last = idx_i
            continue
        selected.append(idx_i)
        last = idx_i

    max_events = int(max(1, round((float(x.size) / float(sample_rate)) * float(max_events_per_sec))))
    if len(selected) > max_events:
        selected = sorted(sorted(selected, key=lambda idx: float(score[idx]), reverse=True)[:max_events])
    if not selected:
        return x

    low_hz = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_LOW_HZ",
        90.0,
        low=20.0,
        high=float(sample_rate) * 0.5 - 2.0,
        profile_defaults=profile_defaults,
    )
    high_hz = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_HIGH_HZ",
        950.0,
        low=float(low_hz) + 1.0,
        high=float(sample_rate) * 0.5 - 1.0,
        profile_defaults=profile_defaults,
    )
    edge_hz = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_BILABIAL_EDGE_HZ",
        160.0,
        low=0.0,
        high=1200.0,
        profile_defaults=profile_defaults,
    )
    low_band = _spectral_band(x, sample_rate, low_hz, high_hz, edge_hz)

    out = x.astype(np.float32, copy=True)
    pre_samples = max(1, int(round(float(sample_rate) * float(pre_close_ms) / 1000.0)))
    puff_samples = max(1, int(round(float(sample_rate) * float(puff_ms) / 1000.0)))
    for idx in selected:
        idx_i = int(idx)
        rel = min(1.6, float(score[idx_i]) / float(cutoff + 1e-6))
        close_amt = min(0.9, float(close_strength) * rel)
        release_amt = min(1.0, float(release_strength) * rel)

        close_start = max(0, idx_i - pre_samples)
        if idx_i > close_start and close_amt > 1e-6:
            close_len = int(idx_i - close_start)
            phase = np.linspace(0.0, np.pi, close_len, endpoint=False, dtype=np.float32)
            ramp = 0.5 - 0.5 * np.cos(phase)
            out[close_start:idx_i] *= (1.0 - close_amt * ramp).astype(np.float32, copy=False)

        release_end = min(int(out.size), idx_i + puff_samples)
        if release_end > idx_i and release_amt > 1e-6:
            rel_len = int(release_end - idx_i)
            phase = np.linspace(0.0, np.pi, rel_len, endpoint=False, dtype=np.float32)
            bump = np.sin(phase).astype(np.float32, copy=False)
            source = x[idx_i:release_end] + low_band[idx_i:release_end] * 1.35
            out[idx_i:release_end] += source * (release_amt * bump)

    return out.astype(np.float32, copy=False)


def _prepare_audio_for_model(audio_input, *, sample_rate=16000):
    """Optional model-only audio shaping before Wav2Vec.

    This intentionally does not affect the final playback audio. It only makes
    the conditioning signal more or less assertive for the avatar model.
    """
    x = np.asarray(audio_input, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if x.size <= 0:
        return x

    global _AUDIO_PROFILE_LOGGED
    profile_name = str(os.getenv("LIVEAVATAR_MODEL_AUDIO_PROFILE", "custom") or "custom").strip().lower()
    profile_defaults = _audio_profile_defaults()
    if not _AUDIO_PROFILE_LOGGED:
        _AUDIO_PROFILE_LOGGED = True
        logging.info(
            "LiveAvatar model audio preprocessing: profile=%s clarity_chain=%s sample_rate=%s",
            profile_name,
            str(profile_defaults.get("LIVEAVATAR_MODEL_AUDIO_CLARITY_CHAIN", os.getenv("LIVEAVATAR_MODEL_AUDIO_CLARITY_CHAIN", "0"))),
            str(sample_rate),
        )
    advance_ms = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_ADVANCE_MS",
        0.0,
        low=-200.0,
        high=200.0,
        profile_defaults=profile_defaults,
    )
    if abs(advance_ms) >= 0.5 and sample_rate > 0:
        shift = int(round(float(sample_rate) * float(advance_ms) / 1000.0))
        if shift > 0 and shift < x.size:
            x = np.concatenate((x[shift:], np.zeros((shift,), dtype=np.float32))).astype(np.float32, copy=False)
        elif shift < 0 and -shift < x.size:
            n = int(-shift)
            x = np.concatenate((np.zeros((n,), dtype=np.float32), x[:-n])).astype(np.float32, copy=False)
    original_x = x.astype(np.float32, copy=True)

    preemphasis = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_PREEMPHASIS",
        0.0,
        low=0.0,
        high=0.98,
        profile_defaults=profile_defaults,
    )
    if preemphasis > 0.0 and x.size > 1:
        y = np.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - float(preemphasis) * x[:-1]
        x = y

    target_rms = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_TARGET_RMS",
        0.0,
        low=0.0,
        high=0.5,
        profile_defaults=profile_defaults,
    )
    max_gain = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_MAX_GAIN",
        2.0,
        low=0.1,
        high=8.0,
        profile_defaults=profile_defaults,
    )
    gain = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_GAIN",
        1.0,
        low=0.1,
        high=8.0,
        profile_defaults=profile_defaults,
    )
    if target_rms > 0.0:
        rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float32))) + 1e-8)
        if rms > 0.0:
            gain *= min(float(max_gain), float(target_rms) / rms)
    if abs(gain - 1.0) > 1e-6:
        x = x * float(gain)

    transient_boost = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_BOOST",
        0.0,
        low=0.0,
        high=2.0,
        profile_defaults=profile_defaults,
    )
    if transient_boost > 1e-6 and x.size > 4 and sample_rate > 0:
        fast_ms = _env_float(
            "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_FAST_MS",
            18.0,
            low=2.0,
            high=80.0,
            profile_defaults=profile_defaults,
        )
        slow_ms = _env_float(
            "LIVEAVATAR_MODEL_AUDIO_TRANSIENT_SLOW_MS",
            120.0,
            low=max(4.0, float(fast_ms) + 1.0),
            high=500.0,
            profile_defaults=profile_defaults,
        )
        fast_env = _smooth_abs_envelope(x, sample_rate, fast_ms)
        slow_env = _smooth_abs_envelope(x, sample_rate, slow_ms)
        transient = np.maximum(0.0, fast_env - slow_env) / (slow_env + 1e-4)
        boost = 1.0 + float(transient_boost) * np.clip(transient, 0.0, 2.0)
        x = x * boost.astype(np.float32, copy=False)

    quiet_expand = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_QUIET_EXPAND",
        0.0,
        low=0.0,
        high=4.0,
        profile_defaults=profile_defaults,
    )
    quiet_threshold = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_QUIET_THRESHOLD",
        0.0,
        low=0.0,
        high=0.5,
        profile_defaults=profile_defaults,
    )
    if quiet_expand > 1e-6 and quiet_threshold > 1e-6 and x.size > 4 and sample_rate > 0:
        floor = _env_float(
            "LIVEAVATAR_MODEL_AUDIO_QUIET_FLOOR",
            0.18,
            low=0.0,
            high=1.0,
            profile_defaults=profile_defaults,
        )
        env = _smooth_abs_envelope(x, sample_rate, 35.0)
        ratio = np.clip(env / float(quiet_threshold), 0.0, 1.0)
        attenuation = float(floor) + float(1.0 - floor) * np.power(ratio, float(quiet_expand))
        x = x * np.where(env < float(quiet_threshold), attenuation, 1.0).astype(np.float32, copy=False)

    x = _limited_band_boost(
        x,
        sample_rate,
        profile_defaults,
        prefix="VOWEL",
        low_default=180.0,
        high_default=950.0,
        soft_default=0.16,
        loud_default=0.33,
    )
    x = _limited_band_boost(
        x,
        sample_rate,
        profile_defaults,
        prefix="FORMANT",
        low_default=850.0,
        high_default=2300.0,
        soft_default=0.14,
        loud_default=0.29,
    )
    x = _limited_band_boost(
        x,
        sample_rate,
        profile_defaults,
        prefix="PRESENCE",
        low_default=650.0,
        high_default=3600.0,
        soft_default=0.13,
        loud_default=0.27,
    )
    x = _limited_band_boost(
        x,
        sample_rate,
        profile_defaults,
        prefix="CONSONANT",
        low_default=2400.0,
        high_default=6200.0,
        soft_default=0.10,
        loud_default=0.22,
    )
    x = _apply_avatar_clarity_chain(x, sample_rate, profile_defaults)
    x = _apply_bilabial_puff(x, sample_rate, profile_defaults)

    if _env_flag("LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP", "0", profile_defaults=profile_defaults):
        drive = _env_float(
            "LIVEAVATAR_MODEL_AUDIO_SOFT_CLIP_DRIVE",
            1.25,
            low=0.1,
            high=8.0,
            profile_defaults=profile_defaults,
        )
        x = np.tanh(x * float(drive)) / float(max(1e-6, drive))

    original_mix = _env_float(
        "LIVEAVATAR_MODEL_AUDIO_ORIGINAL_MIX",
        0.0,
        low=0.0,
        high=1.0,
        profile_defaults=profile_defaults,
    )
    if original_mix > 1e-6:
        x = x * float(1.0 - original_mix) + original_x * float(original_mix)

    return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)


class AudioEncoder():

    def __init__(self, device='cpu', model_id="facebook/wav2vec2-base-960h"):
        # load pretrained model
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)

        self.model = self.model.to(device)

        self.video_rate = 30

    def extract_audio_feat(self,
                           audio_path,
                           return_all_layers=False,
                           dtype=torch.float32):
        audio_input, sample_rate = librosa.load(audio_path, sr=16000)
        audio_input = _prepare_audio_for_model(audio_input, sample_rate=sample_rate)

        input_values = self.processor(
            audio_input, sampling_rate=sample_rate,
            return_tensors="pt").input_values

        # INFERENCE

        # retrieve logits & take argmax
        res = self.model(
            input_values.to(device=self.model.device, dtype=self.model.dtype), output_hidden_states=True)
        if return_all_layers:
            feat = torch.cat(res.hidden_states)
        else:
            feat = res.hidden_states[-1]
        feat = linear_interpolation(
            feat, input_fps=50, output_fps=self.video_rate)

        z = feat.to(dtype)  # Encoding for the motion
        return z

    def extract_audio_feat_from_array(self,
                                       audio_array,
                                       sample_rate=16000,
                                       return_all_layers=False,
                                       dtype=torch.float32):
        """从 numpy array 提取音频特征（不需要文件路径）"""
        # 直接使用传入的 audio_array，不需要 librosa.load
        audio_input = _prepare_audio_for_model(audio_array, sample_rate=sample_rate)

        input_values = self.processor(
            audio_input, sampling_rate=sample_rate,
            return_tensors="pt").input_values

        # INFERENCE

        # retrieve logits & take argmax
        res = self.model(
            input_values.to(device=self.model.device, dtype=torch.bfloat16), output_hidden_states=True)
        if return_all_layers:
            feat = torch.cat(res.hidden_states)
        else:
            feat = res.hidden_states[-1]
        feat = linear_interpolation(
            feat, input_fps=50, output_fps=self.video_rate)

        z = feat.to(dtype)  # Encoding for the motion
        return z

    def get_audio_embed_bucket(self,
                               audio_embed,
                               stride=2,
                               batch_frames=12,
                               m=2):
        num_layers, audio_frame_num, audio_dim = audio_embed.shape

        if num_layers > 1:
            return_all_layers = True
        else:
            return_all_layers = False

        min_batch_num = int(audio_frame_num / (batch_frames * stride)) + 1

        bucket_num = min_batch_num * batch_frames
        batch_idx = [stride * i for i in range(bucket_num)]
        batch_audio_eb = []
        for bi in batch_idx:
            if bi < audio_frame_num:
                audio_sample_stride = 2
                chosen_idx = list(
                    range(bi - m * audio_sample_stride,
                          bi + (m + 1) * audio_sample_stride,
                          audio_sample_stride))
                chosen_idx = [0 if c < 0 else c for c in chosen_idx]
                chosen_idx = [
                    audio_frame_num - 1 if c >= audio_frame_num else c
                    for c in chosen_idx
                ]

                if return_all_layers:
                    frame_audio_embed = audio_embed[:, chosen_idx].flatten(
                        start_dim=-2, end_dim=-1)
                else:
                    frame_audio_embed = audio_embed[0][chosen_idx].flatten()
            else:
                frame_audio_embed = \
                torch.zeros([audio_dim * (2 * m + 1)], device=audio_embed.device) if not return_all_layers \
                    else torch.zeros([num_layers, audio_dim * (2 * m + 1)], device=audio_embed.device)
            batch_audio_eb.append(frame_audio_embed)
        batch_audio_eb = torch.cat([c.unsqueeze(0) for c in batch_audio_eb],
                                   dim=0)

        return batch_audio_eb, min_batch_num

    def get_audio_embed_bucket_fps(self,
                                   audio_embed,
                                   fps=16,
                                   batch_frames=81,
                                   m=0):
        num_layers, audio_frame_num, audio_dim = audio_embed.shape

        if num_layers > 1:
            return_all_layers = True
        else:
            return_all_layers = False

        scale = self.video_rate / fps

        min_batch_num = int(audio_frame_num / (batch_frames * scale)) + 1

        bucket_num = min_batch_num * batch_frames
        padd_audio_num = math.ceil(min_batch_num * batch_frames / fps *
                                   self.video_rate) - audio_frame_num
        batch_idx = get_sample_indices(
            original_fps=int(self.video_rate),
            total_frames=int(audio_frame_num + padd_audio_num),
            target_fps=int(fps),
            num_sample=int(bucket_num),
            fixed_start=0)
        batch_audio_eb = []
        audio_sample_stride = int(self.video_rate / fps)
        for bi in batch_idx:
            if bi < audio_frame_num:

                chosen_idx = list(
                    range(bi - m * audio_sample_stride,
                          bi + (m + 1) * audio_sample_stride,
                          audio_sample_stride))
                chosen_idx = [0 if c < 0 else c for c in chosen_idx]
                chosen_idx = [
                    audio_frame_num - 1 if c >= audio_frame_num else c
                    for c in chosen_idx
                ]

                if return_all_layers:
                    frame_audio_embed = audio_embed[:, chosen_idx].flatten(
                        start_dim=-2, end_dim=-1)
                else:
                    frame_audio_embed = audio_embed[0][chosen_idx].flatten()
            else:
                frame_audio_embed = \
                torch.zeros([audio_dim * (2 * m + 1)], device=audio_embed.device) if not return_all_layers \
                    else torch.zeros([num_layers, audio_dim * (2 * m + 1)], device=audio_embed.device)
            batch_audio_eb.append(frame_audio_embed)
        batch_audio_eb = torch.cat([c.unsqueeze(0) for c in batch_audio_eb],
                                   dim=0)

        return batch_audio_eb, min_batch_num
