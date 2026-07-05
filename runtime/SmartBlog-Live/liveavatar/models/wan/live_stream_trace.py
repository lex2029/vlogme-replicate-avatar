from __future__ import annotations

import threading
import time


class LiveaudioTrace:
    """
    Minimal runtime telemetry for the first liveaudio reply path.

    It keeps only the timestamps that matter for first-response latency:
    first encoded chunk, first clip selection, first block start,
    first raw enqueue and first raw write.
    """

    def __init__(
        self,
        *,
        rank: int,
        trace_t0: float,
        trace_tag: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.rank = int(rank)
        self.trace_t0 = float(trace_t0)
        self.trace_tag = str(trace_tag or "").strip()
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._first_chunk_dt: float | None = None
        self._first_clip_dt: float | None = None
        self._first_block_dt: float | None = None
        self._first_raw_enqueue_dt: float | None = None
        self._first_raw_written_dt: float | None = None

    def _dt(self) -> float:
        return float(time.perf_counter() - self.trace_t0)

    @staticmethod
    def _fmt_delta(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{float(value):.3f}s"

    def _emit(self, message: str) -> None:
        if not bool(self.enabled):
            return
        if self.trace_tag:
            print(f"{message} tag={self.trace_tag}", flush=True)
        else:
            print(message, flush=True)

    def note_chunk_loaded(
        self,
        *,
        chunk_idx: int,
        clips_added: int,
        expected_frames: int,
        enc_src: str,
        enc_dt: float,
        queue_depth: int,
        seen_chunks: int,
    ) -> None:
        now_dt = self._dt()
        first_chunk = False
        with self._lock:
            if self._first_chunk_dt is None:
                self._first_chunk_dt = float(now_dt)
                first_chunk = True
        self._emit(
            f"Rank {self.rank}: liveaudio loaded chunk={int(chunk_idx)} "
            f"clips_added={int(clips_added)} exp_frames={int(expected_frames)} "
            f"enc_src={str(enc_src)} enc_dt={float(enc_dt):.3f}s "
            f"q={int(queue_depth)} seen_chunks={int(seen_chunks)} dt={float(now_dt):.3f}s"
        )
        if first_chunk:
            self._emit(
                f"TPP first-chunk-loaded rank={self.rank} dt={float(now_dt):.3f}s "
                f"chunk={int(chunk_idx)} q={int(queue_depth)} seen_chunks={int(seen_chunks)}"
            )

    def note_first_clip(
        self,
        *,
        event: str,
        q_after: int,
        is_silence: bool,
        seen_chunks: int,
        clip_shape: tuple[int, ...] | None = None,
    ) -> None:
        now_dt = self._dt()
        from_chunk = None
        with self._lock:
            if self._first_clip_dt is None:
                self._first_clip_dt = float(now_dt)
            if self._first_chunk_dt is not None:
                from_chunk = float(now_dt - self._first_chunk_dt)
        shape_suffix = ""
        if clip_shape:
            shape_suffix = f" shape={list(clip_shape)}"
        self._emit(
            f"TPP first-clip-{str(event)} rank={self.rank} dt={float(now_dt):.3f}s "
            f"from_first_chunk={self._fmt_delta(from_chunk)} q={int(q_after)} "
            f"is_silence={1 if bool(is_silence) else 0} seen_chunks={int(seen_chunks)}{shape_suffix}"
        )

    def note_first_block_start(self, *, q_depth: int, done: bool) -> None:
        now_dt = self._dt()
        from_clip = None
        from_chunk = None
        with self._lock:
            if self._first_block_dt is None:
                self._first_block_dt = float(now_dt)
            if self._first_clip_dt is not None:
                from_clip = float(now_dt - self._first_clip_dt)
            if self._first_chunk_dt is not None:
                from_chunk = float(now_dt - self._first_chunk_dt)
        self._emit(
            f"TPP first-block-start rank={self.rank} dt={float(now_dt):.3f}s "
            f"from_first_clip={self._fmt_delta(from_clip)} "
            f"from_first_chunk={self._fmt_delta(from_chunk)} "
            f"q={int(q_depth)} done={1 if bool(done) else 0}"
        )

    def note_first_block_setup(self, *, setup_dt: float, q_depth: int, done: bool) -> None:
        now_dt = self._dt()
        from_block = None
        with self._lock:
            if self._first_block_dt is not None:
                from_block = float(now_dt - self._first_block_dt)
        self._emit(
            f"TPP first-block-setup rank={self.rank} dt={float(now_dt):.3f}s "
            f"from_first_block={self._fmt_delta(from_block)} "
            f"setup={float(setup_dt):.3f}s q={int(q_depth)} done={1 if bool(done) else 0}"
        )

    def note_first_block_step_phase(
        self,
        *,
        step_idx: int,
        total_steps: int,
        phase: str,
        phase_dt: float,
        q_depth: int,
        done: bool,
    ) -> None:
        now_dt = self._dt()
        from_block = None
        with self._lock:
            if self._first_block_dt is not None:
                from_block = float(now_dt - self._first_block_dt)
        self._emit(
            f"TPP first-block-step-phase rank={self.rank} "
            f"step={int(step_idx)}/{int(total_steps)} phase={str(phase)} "
            f"dt={float(now_dt):.3f}s from_first_block={self._fmt_delta(from_block)} "
            f"phase_dt={float(phase_dt):.3f}s q={int(q_depth)} done={1 if bool(done) else 0}"
        )

    def note_first_clip_prepare(self, *, prep_dt: float, q_depth: int, done: bool) -> None:
        now_dt = self._dt()
        from_clip = None
        with self._lock:
            if self._first_clip_dt is not None:
                from_clip = float(now_dt - self._first_clip_dt)
        self._emit(
            f"TPP first-clip-prepare rank={self.rank} dt={float(now_dt):.3f}s "
            f"from_first_clip={self._fmt_delta(from_clip)} "
            f"prep={float(prep_dt):.3f}s q={int(q_depth)} done={1 if bool(done) else 0}"
        )

    def note_first_prefill(self, *, prefill_dt: float, q_depth: int, done: bool) -> None:
        now_dt = self._dt()
        from_clip = None
        with self._lock:
            if self._first_clip_dt is not None:
                from_clip = float(now_dt - self._first_clip_dt)
        self._emit(
            f"TPP first-prefill rank={self.rank} dt={float(now_dt):.3f}s "
            f"from_first_clip={self._fmt_delta(from_clip)} "
            f"prefill={float(prefill_dt):.3f}s q={int(q_depth)} done={1 if bool(done) else 0}"
        )

    def note_first_prefill_phase(
        self,
        *,
        phase: str,
        phase_dt: float,
        q_depth: int,
        done: bool,
    ) -> None:
        now_dt = self._dt()
        from_clip = None
        with self._lock:
            if self._first_clip_dt is not None:
                from_clip = float(now_dt - self._first_clip_dt)
        self._emit(
            f"TPP first-prefill-phase rank={self.rank} phase={str(phase)} "
            f"dt={float(now_dt):.3f}s from_first_clip={self._fmt_delta(from_clip)} "
            f"phase_dt={float(phase_dt):.3f}s q={int(q_depth)} done={1 if bool(done) else 0}"
        )

    def note_first_raw_enqueue(
        self,
        *,
        frames_enqueued: int,
        backlog_before: int,
        backlog_after: int,
        q_depth: int,
        done: bool,
    ) -> None:
        now_dt = self._dt()
        from_block = None
        from_chunk = None
        should_emit = False
        with self._lock:
            if self._first_raw_enqueue_dt is None:
                self._first_raw_enqueue_dt = float(now_dt)
                should_emit = True
            if self._first_block_dt is not None:
                from_block = float(now_dt - self._first_block_dt)
            if self._first_chunk_dt is not None:
                from_chunk = float(now_dt - self._first_chunk_dt)
        if not should_emit:
            return
        self._emit(
            f"TPP first-raw-enqueue rank={self.rank} dt={float(now_dt):.3f}s "
            f"from_first_block={self._fmt_delta(from_block)} "
            f"from_first_chunk={self._fmt_delta(from_chunk)} "
            f"frames={int(frames_enqueued)} backlog={int(backlog_before)}->{int(backlog_after)} "
            f"q={int(q_depth)} done={1 if bool(done) else 0}"
        )

    def note_first_raw_written(self, *, frames_streamed: int, backlog_bytes: int) -> None:
        now_dt = self._dt()
        from_enqueue = None
        from_chunk = None
        should_emit = False
        with self._lock:
            if self._first_raw_written_dt is None:
                self._first_raw_written_dt = float(now_dt)
                should_emit = True
            if self._first_raw_enqueue_dt is not None:
                from_enqueue = float(now_dt - self._first_raw_enqueue_dt)
            if self._first_chunk_dt is not None:
                from_chunk = float(now_dt - self._first_chunk_dt)
        if not should_emit:
            return
        self._emit(
            f"TPP first-raw-written rank={self.rank} dt={float(now_dt):.3f}s "
            f"from_first_raw_enqueue={self._fmt_delta(from_enqueue)} "
            f"from_first_chunk={self._fmt_delta(from_chunk)} "
            f"frames_streamed={int(frames_streamed)} backlog_bytes={int(backlog_bytes)}"
        )
