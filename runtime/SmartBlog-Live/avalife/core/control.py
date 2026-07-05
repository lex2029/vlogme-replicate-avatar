from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.distributed as dist


CONTROL_PAYLOAD_LEN = 29


@dataclass(frozen=True)
class ControlCommand:
    command: str | None
    prompt: str | None = None
    image_path: str | None = None
    audio_path: str | None = None
    num_clip: int | None = None
    sample_steps: int | None = None
    sample_guide_scale: float | None = None
    infer_frames: int | None = None
    size: str | None = None
    base_seed: int | None = None
    sample_solver: str | None = None
    live_raw_dir: str | None = None
    face_restore: float | None = None
    background_restore: float | None = None
    load_lora: bool | None = None
    job_id: str | None = None
    enable_live_hls: bool | None = None
    lora_path: str | None = None
    save_live_raw_mp4: bool | None = None
    video_prompt: str | None = None
    negative_prompt: str | None = None
    idle_prompt: str | None = None
    stream_file_output_path: str | None = None
    stream_file_output_width: int | None = None
    stream_file_output_height: int | None = None
    stream_file_output_fps: float | None = None
    stream_file_trim_duration_sec: float | None = None
    stream_file_interpolation: str | None = None
    tpp_cfg_mode: str | None = None
    lipsync_audio_path: str | None = None


def _payload_from_command(cmd: ControlCommand) -> list[Any]:
    return [
        cmd.command,
        cmd.prompt,
        cmd.image_path,
        cmd.audio_path,
        cmd.num_clip,
        cmd.sample_steps,
        cmd.sample_guide_scale,
        cmd.infer_frames,
        cmd.size,
        cmd.base_seed,
        cmd.sample_solver,
        cmd.live_raw_dir if cmd.command == "infer" else cmd.lora_path,
        cmd.load_lora,
        cmd.job_id,
        cmd.enable_live_hls,
        cmd.face_restore,
        cmd.background_restore,
        cmd.save_live_raw_mp4,
        cmd.video_prompt,
        cmd.negative_prompt,
        cmd.idle_prompt,
        cmd.stream_file_output_path,
        cmd.stream_file_output_width,
        cmd.stream_file_output_height,
        cmd.stream_file_output_fps,
        cmd.stream_file_trim_duration_sec,
        cmd.stream_file_interpolation,
        cmd.tpp_cfg_mode,
        cmd.lipsync_audio_path,
    ]


def _command_from_payload(payload: list[Any]) -> ControlCommand:
    def item(index: int) -> Any:
        return payload[index] if index < len(payload) else None

    return ControlCommand(
        command=item(0),
        prompt=item(1),
        image_path=item(2),
        audio_path=item(3),
        num_clip=item(4),
        sample_steps=item(5),
        sample_guide_scale=item(6),
        infer_frames=item(7),
        size=item(8),
        base_seed=item(9),
        sample_solver=item(10),
        live_raw_dir=item(11) if item(0) == "infer" else None,
        load_lora=item(12),
        job_id=item(13),
        enable_live_hls=item(14),
        face_restore=item(15),
        background_restore=item(16),
        save_live_raw_mp4=item(17),
        video_prompt=item(18) if len(payload) > 18 else None,
        negative_prompt=item(19) if len(payload) > 19 else None,
        idle_prompt=item(20) if len(payload) > 20 else None,
        stream_file_output_path=item(21) if len(payload) > 21 else None,
        stream_file_output_width=item(22) if len(payload) > 22 else None,
        stream_file_output_height=item(23) if len(payload) > 23 else None,
        stream_file_output_fps=item(24) if len(payload) > 24 else None,
        stream_file_trim_duration_sec=item(25) if len(payload) > 25 else None,
        stream_file_interpolation=item(26) if len(payload) > 26 else None,
        tpp_cfg_mode=item(27) if len(payload) > 27 else None,
        lipsync_audio_path=item(28) if len(payload) > 28 else None,
        lora_path=item(11) if item(0) == "reload" else None,
    )


def broadcast_command(*, cmd: ControlCommand, src_rank: int, control_group: Any) -> None:
    payload = _payload_from_command(cmd)
    dist.broadcast_object_list(payload, src=int(src_rank), group=control_group)


def receive_command(*, src_rank: int, control_group: Any) -> ControlCommand:
    payload: list[Any] = [None] * CONTROL_PAYLOAD_LEN
    dist.broadcast_object_list(payload, src=int(src_rank), group=control_group)
    return _command_from_payload(payload)


def build_infer_command(
    *,
    prompt: str,
    image_path: str,
    audio_path: str,
    num_clip: int,
    sample_steps: int,
    sample_guide_scale: float,
    infer_frames: int,
    size: str,
    base_seed: int,
    sample_solver: str,
    live_raw_dir: str | None,
    face_restore: float,
    background_restore: float,
    job_id: str | None,
    enable_live_hls: bool,
    save_live_raw_mp4: bool = False,
    video_prompt: str = "",
    negative_prompt: str = "",
    idle_prompt: str = "",
    stream_file_output_path: str = "",
    stream_file_output_width: int = 0,
    stream_file_output_height: int = 0,
    stream_file_output_fps: float = 0.0,
    stream_file_trim_duration_sec: float = 0.0,
    stream_file_interpolation: str = "",
    tpp_cfg_mode: str = "",
    lipsync_audio_path: str = "",
) -> ControlCommand:
    return ControlCommand(
        command="infer",
        prompt=str(prompt),
        image_path=str(image_path),
        audio_path=str(audio_path),
        num_clip=int(num_clip),
        sample_steps=int(sample_steps),
        sample_guide_scale=float(sample_guide_scale),
        infer_frames=int(infer_frames),
        size=str(size),
        base_seed=int(base_seed),
        sample_solver=str(sample_solver),
        live_raw_dir=(str(live_raw_dir) if live_raw_dir else None),
        face_restore=float(face_restore),
        background_restore=float(background_restore),
        job_id=(str(job_id) if job_id is not None else None),
        enable_live_hls=bool(enable_live_hls),
        save_live_raw_mp4=bool(save_live_raw_mp4),
        video_prompt=str(video_prompt or ""),
        negative_prompt=str(negative_prompt or ""),
        idle_prompt=str(idle_prompt or ""),
        stream_file_output_path=str(stream_file_output_path or ""),
        stream_file_output_width=int(stream_file_output_width or 0),
        stream_file_output_height=int(stream_file_output_height or 0),
        stream_file_output_fps=float(stream_file_output_fps or 0.0),
        stream_file_trim_duration_sec=float(stream_file_trim_duration_sec or 0.0),
        stream_file_interpolation=str(stream_file_interpolation or ""),
        tpp_cfg_mode=str(tpp_cfg_mode or ""),
        lipsync_audio_path=str(lipsync_audio_path or ""),
    )


def build_reload_command(*, load_lora: bool, lora_path: str | None) -> ControlCommand:
    return ControlCommand(
        command="reload",
        load_lora=bool(load_lora),
        lora_path=(str(lora_path) if lora_path else None),
    )


def build_idle_command() -> ControlCommand:
    return ControlCommand(command="idle")
