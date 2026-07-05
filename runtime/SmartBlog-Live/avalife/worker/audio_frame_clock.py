from __future__ import annotations

import math


def nominal_samples_per_video_frame(*, sample_rate: int, fps: int) -> int:
    sr = int(max(1, int(sample_rate)))
    fps_i = int(max(1, int(fps)))
    return int(max(1, sr // fps_i))


def max_samples_per_video_frame(*, sample_rate: int, fps: int) -> int:
    sr = int(max(1, int(sample_rate)))
    fps_i = int(max(1, int(fps)))
    return int(max(1, int(math.ceil(float(sr) / float(fps_i)))))


def samples_for_video_frame_index(*, sample_rate: int, fps: int, frame_index: int) -> int:
    sr = int(max(1, int(sample_rate)))
    fps_i = int(max(1, int(fps)))
    idx = int(max(0, int(frame_index)))
    start = int((idx * sr) // fps_i)
    end = int(((idx + 1) * sr) // fps_i)
    return int(max(1, int(end - start)))


def video_frames_for_audio_samples(*, sample_count: int, sample_rate: int, fps: int) -> int:
    samples = int(max(0, int(sample_count or 0)))
    if samples <= 0:
        return 0
    sr = int(max(1, int(sample_rate)))
    fps_i = int(max(1, int(fps)))
    frames = int(math.ceil(float(samples * fps_i) / float(sr)))
    while int((frames * sr) // fps_i) < int(samples):
        frames += 1
    return int(max(1, frames))


class AudioFrameClock:
    def __init__(self, *, sample_rate: int, fps: int) -> None:
        self.sample_rate = int(max(1, int(sample_rate)))
        self.fps = int(max(1, int(fps)))
        self.frame_index = 0

    def reset(self) -> None:
        self.frame_index = 0

    def samples_for_next_frame(self) -> int:
        return samples_for_video_frame_index(
            sample_rate=int(self.sample_rate),
            fps=int(self.fps),
            frame_index=int(self.frame_index),
        )

    def bytes_for_next_frame(self) -> int:
        return int(self.samples_for_next_frame() * 2)

    def advance(self) -> int:
        samples = int(self.samples_for_next_frame())
        self.frame_index = int(self.frame_index + 1)
        return int(samples)
