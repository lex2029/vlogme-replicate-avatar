from __future__ import annotations

import struct
from typing import Any


LIVE_RAW_SHM_MAGIC = b"AVLRSHM1"
LIVE_RAW_SHM_VERSION = 1
LIVE_RAW_SHM_HEADER_STRUCT = struct.Struct("<8sIIQQQQQQQBB6x")
LIVE_RAW_SHM_HEADER_BYTES = int(LIVE_RAW_SHM_HEADER_STRUCT.size)

_PROMPT_MODE_TO_CODE = {
    "speech": 1,
    "idle": 2,
}
_PROMPT_CODE_TO_MODE = {
    1: "speech",
    2: "idle",
}


def live_raw_shm_total_bytes(*, frame_bytes: int, frame_capacity: int) -> int:
    fb = int(max(1, int(frame_bytes)))
    cap = int(max(1, int(frame_capacity)))
    return int(LIVE_RAW_SHM_HEADER_BYTES + (fb * cap))


def live_raw_shm_frame_region(shm_buf: memoryview) -> memoryview:
    return shm_buf[int(LIVE_RAW_SHM_HEADER_BYTES) :]


def live_raw_shm_write_header(
    shm_buf: memoryview,
    *,
    written_frames: int,
    enqueued_frames: int,
    backlog_bytes: int,
    prompt_mode: str,
    mode_seq: int,
    mode_start_frame: int,
    source_chunk_idx: int,
    source_chunk_start_frame: int,
    done: bool,
) -> None:
    prompt_mode_norm = str(prompt_mode or "speech").strip().lower()
    if prompt_mode_norm not in _PROMPT_MODE_TO_CODE:
        prompt_mode_norm = "speech"
    LIVE_RAW_SHM_HEADER_STRUCT.pack_into(
        shm_buf,
        0,
        LIVE_RAW_SHM_MAGIC,
        int(LIVE_RAW_SHM_VERSION),
        int(LIVE_RAW_SHM_HEADER_BYTES),
        int(max(0, int(written_frames))),
        int(max(0, int(enqueued_frames))),
        int(max(0, int(backlog_bytes))),
        int(max(0, int(mode_seq))),
        int(max(0, int(mode_start_frame))),
        int(max(0, int(source_chunk_idx))),
        int(max(0, int(source_chunk_start_frame))),
        int(_PROMPT_MODE_TO_CODE[prompt_mode_norm]),
        1 if bool(done) else 0,
    )


def live_raw_shm_read_header(shm_buf: memoryview) -> dict[str, Any] | None:
    if len(shm_buf) < int(LIVE_RAW_SHM_HEADER_BYTES):
        return None
    try:
        (
            magic,
            version,
            header_bytes,
            written_frames,
            enqueued_frames,
            backlog_bytes,
            mode_seq,
            mode_start_frame,
            source_chunk_idx,
            source_chunk_start_frame,
            prompt_mode_code,
            done_flag,
        ) = LIVE_RAW_SHM_HEADER_STRUCT.unpack_from(shm_buf, 0)
    except Exception:
        return None
    if bytes(magic) != bytes(LIVE_RAW_SHM_MAGIC):
        return None
    if int(version) != int(LIVE_RAW_SHM_VERSION):
        return None
    if int(header_bytes) < int(LIVE_RAW_SHM_HEADER_BYTES):
        return None
    prompt_mode = _PROMPT_CODE_TO_MODE.get(int(prompt_mode_code), "speech")
    return {
        "written_frames": int(max(0, int(written_frames))),
        "enqueued_frames": int(max(0, int(enqueued_frames))),
        "backlog_bytes": int(max(0, int(backlog_bytes))),
        "prompt_mode": str(prompt_mode),
        "mode_seq": int(max(0, int(mode_seq))),
        "mode_start_frame": int(max(0, int(mode_start_frame))),
        "source_chunk_idx": int(max(0, int(source_chunk_idx))),
        "source_chunk_start_frame": int(max(0, int(source_chunk_start_frame))),
        "done": bool(done_flag),
        "transport": "shm_ring",
        "header_bytes": int(max(0, int(header_bytes))),
    }
