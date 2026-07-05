from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class InferRequest:
    prompt: str
    image_path: str
    audio_path: str
    num_clip: int
    sample_steps: int
    sample_guide_scale: float
    infer_frames: int
    size: str
    base_seed: int
    sample_solver: str
    face_restore: float = 0.0
    background_restore: float = 0.0
    job_id: str | None = None
    enable_live_hls: bool = False
    live_raw_dir: str | None = None
    save_live_raw_mp4: bool = False
    video_prompt: str = ""
    negative_prompt: str = ""
    idle_prompt: str = ""
    stream_file_output_path: str = ""
    stream_file_output_width: int = 0
    stream_file_output_height: int = 0
    stream_file_output_fps: float = 0.0
    stream_file_trim_duration_sec: float = 0.0
    stream_file_interpolation: str = ""
    tpp_cfg_mode: str = ""
    lipsync_audio_path: str = ""

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "InferRequest":
        return cls(
            prompt=str(payload.get("prompt") or ""),
            image_path=str(payload.get("image_path") or ""),
            audio_path=str(payload.get("audio_path") or ""),
            lipsync_audio_path=str(payload.get("lipsync_audio_path") or ""),
            num_clip=int(payload.get("num_clip") or 0),
            sample_steps=int(payload.get("sample_steps") or 0),
            sample_guide_scale=float(payload.get("sample_guide_scale") or 0.0),
            infer_frames=int(payload.get("infer_frames") or 0),
            size=str(payload.get("size") or ""),
            base_seed=int(payload.get("base_seed") or 0),
            sample_solver=str(payload.get("sample_solver") or ""),
            face_restore=float(payload.get("face_restore") or 0.0),
            background_restore=float(payload.get("background_restore") or 0.0),
            job_id=(str(payload.get("job_id")) if payload.get("job_id") is not None else None),
            enable_live_hls=bool(payload.get("enable_live_hls")),
            live_raw_dir=(str(payload.get("live_raw_dir")) if payload.get("live_raw_dir") else None),
            save_live_raw_mp4=bool(payload.get("save_live_raw_mp4")),
            video_prompt=str(payload.get("video_prompt") or ""),
            negative_prompt=str(payload.get("negative_prompt") or ""),
            idle_prompt=str(payload.get("idle_prompt") or ""),
            stream_file_output_path=str(payload.get("stream_file_output_path") or ""),
            stream_file_output_width=int(payload.get("stream_file_output_width") or 0),
            stream_file_output_height=int(payload.get("stream_file_output_height") or 0),
            stream_file_output_fps=float(payload.get("stream_file_output_fps") or 0.0),
            stream_file_trim_duration_sec=float(payload.get("stream_file_trim_duration_sec") or 0.0),
            stream_file_interpolation=str(payload.get("stream_file_interpolation") or ""),
            tpp_cfg_mode=str(payload.get("tpp_cfg_mode") or ""),
        )


@dataclass(frozen=True)
class InferResponse:
    ok: bool
    video_path: str | None = None
    error: str | None = None
    lock_wait_s: float = 0.0
    run_single_s: float = 0.0
    total_s: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "InferResponse":
        return cls(
            ok=bool(payload.get("ok")),
            video_path=(str(payload.get("video_path")) if payload.get("video_path") else None),
            error=(str(payload.get("error")) if payload.get("error") else None),
            lock_wait_s=float(payload.get("lock_wait_s") or 0.0),
            run_single_s=float(payload.get("run_single_s") or 0.0),
            total_s=float(payload.get("total_s") or 0.0),
        )


@dataclass(frozen=True)
class MediaProcessRequest:
    source_path: str
    source_kind: str
    output_path: str
    output_width: int
    output_height: int
    output_fps: float = 0.0
    preserve_audio: bool = False
    upscale: bool = False
    face_restore: float = 0.0
    background_restore: float = 0.0
    jpeg_quality: int = 95
    trim_duration_sec: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MediaProcessRequest":
        return cls(
            source_path=str(payload.get("source_path") or ""),
            source_kind=str(payload.get("source_kind") or ""),
            output_path=str(payload.get("output_path") or ""),
            output_width=int(payload.get("output_width") or 0),
            output_height=int(payload.get("output_height") or 0),
            output_fps=float(payload.get("output_fps") or 0.0),
            preserve_audio=bool(payload.get("preserve_audio")),
            upscale=bool(payload.get("upscale")),
            face_restore=float(payload.get("face_restore") or 0.0),
            background_restore=float(payload.get("background_restore") or 0.0),
            jpeg_quality=int(payload.get("jpeg_quality") or 95),
            trim_duration_sec=float(payload.get("trim_duration_sec") or 0.0),
        )


@dataclass(frozen=True)
class MediaProcessResponse:
    ok: bool
    output_path: str | None = None
    error: str | None = None
    frames_written: int = 0
    total_s: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MediaProcessResponse":
        return cls(
            ok=bool(payload.get("ok")),
            output_path=(str(payload.get("output_path")) if payload.get("output_path") else None),
            error=(str(payload.get("error")) if payload.get("error") else None),
            frames_written=int(payload.get("frames_written") or 0),
            total_s=float(payload.get("total_s") or 0.0),
        )


@dataclass(frozen=True)
class CancelInferRequest:
    reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CancelInferRequest":
        return cls(
            reason=str(payload.get("reason") or ""),
        )


@dataclass(frozen=True)
class CancelInferResponse:
    ok: bool
    cancelled: bool = False
    active_job_id: str | None = None
    error: str | None = None
    total_s: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CancelInferResponse":
        return cls(
            ok=bool(payload.get("ok")),
            cancelled=bool(payload.get("cancelled")),
            active_job_id=(str(payload.get("active_job_id")) if payload.get("active_job_id") else None),
            error=(str(payload.get("error")) if payload.get("error") else None),
            total_s=float(payload.get("total_s") or 0.0),
        )


@dataclass(frozen=True)
class CancelMediaRequest:
    reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CancelMediaRequest":
        return cls(
            reason=str(payload.get("reason") or ""),
        )


@dataclass(frozen=True)
class CancelMediaResponse:
    ok: bool
    cancelled: bool = False
    active_source_path: str | None = None
    error: str | None = None
    total_s: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CancelMediaResponse":
        return cls(
            ok=bool(payload.get("ok")),
            cancelled=bool(payload.get("cancelled")),
            active_source_path=(str(payload.get("active_source_path")) if payload.get("active_source_path") else None),
            error=(str(payload.get("error")) if payload.get("error") else None),
            total_s=float(payload.get("total_s") or 0.0),
        )


def ping_request() -> dict[str, Any]:
    return {"op": "ping"}


def infer_request(req: InferRequest) -> dict[str, Any]:
    return {"op": "infer", "request": req.to_payload()}


def media_process_request(req: MediaProcessRequest) -> dict[str, Any]:
    return {"op": "media_process", "request": req.to_payload()}


def cancel_infer_request(req: CancelInferRequest | None = None) -> dict[str, Any]:
    req = req or CancelInferRequest()
    return {"op": "cancel_active_infer", "request": req.to_payload()}


def cancel_media_request(req: CancelMediaRequest | None = None) -> dict[str, Any]:
    req = req or CancelMediaRequest()
    return {"op": "cancel_active_media", "request": req.to_payload()}


def ok_response(**extra: Any) -> dict[str, Any]:
    payload = {"ok": True}
    payload.update(extra)
    return payload


def error_response(message: str, **extra: Any) -> dict[str, Any]:
    payload = {"ok": False, "error": str(message or "unknown error")}
    payload.update(extra)
    return payload
