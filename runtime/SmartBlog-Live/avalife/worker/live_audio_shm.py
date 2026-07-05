from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from typing import Any


LIVE_AUDIO_SHM_MAGIC = b"AVLASHM1"
LIVE_AUDIO_SHM_VERSION = 1
LIVE_AUDIO_SHM_HEADER_STRUCT = struct.Struct("<8sIIQQQQQIIB7x")
LIVE_AUDIO_SHM_HEADER_BYTES = int(LIVE_AUDIO_SHM_HEADER_STRUCT.size)

LIVE_AUDIO_CHUNK_META_STRUCT = struct.Struct("<QQIIBBB5x")
LIVE_AUDIO_CHUNK_META_BYTES = int(LIVE_AUDIO_CHUNK_META_STRUCT.size)

_KIND_TO_CODE = {
    "speech": 1,
    "filler": 2,
    "gap_fill": 2,
    "gap-fill": 2,
    "gapfill": 2,
}
_CODE_TO_KIND = {
    1: "speech",
    2: "filler",
}


def attach_shared_memory_no_tracker(name: str) -> shared_memory.SharedMemory:
    shm = shared_memory.SharedMemory(name=str(name), create=False)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    return shm


def live_audio_shm_total_bytes(*, sample_capacity: int, chunk_capacity: int) -> int:
    sample_cap = int(max(1, int(sample_capacity)))
    chunk_cap = int(max(1, int(chunk_capacity)))
    return int(
        LIVE_AUDIO_SHM_HEADER_BYTES
        + (LIVE_AUDIO_CHUNK_META_BYTES * chunk_cap)
        + (sample_cap * 2)
    )


def live_audio_shm_chunk_region(shm_buf: memoryview, *, chunk_capacity: int) -> memoryview:
    chunk_cap = int(max(1, int(chunk_capacity)))
    start = int(LIVE_AUDIO_SHM_HEADER_BYTES)
    end = int(start + (LIVE_AUDIO_CHUNK_META_BYTES * chunk_cap))
    return shm_buf[start:end]


def live_audio_shm_sample_region(shm_buf: memoryview, *, chunk_capacity: int) -> memoryview:
    chunk_cap = int(max(1, int(chunk_capacity)))
    start = int(LIVE_AUDIO_SHM_HEADER_BYTES + (LIVE_AUDIO_CHUNK_META_BYTES * chunk_cap))
    return shm_buf[start:]


def live_audio_shm_write_header(
    shm_buf: memoryview,
    *,
    written_samples: int,
    written_chunks: int,
    total_samples: int,
    speech_end_samples: int,
    sample_capacity: int,
    chunk_capacity: int,
    sample_rate: int,
    done: bool,
) -> None:
    LIVE_AUDIO_SHM_HEADER_STRUCT.pack_into(
        shm_buf,
        0,
        LIVE_AUDIO_SHM_MAGIC,
        int(LIVE_AUDIO_SHM_VERSION),
        int(LIVE_AUDIO_SHM_HEADER_BYTES),
        int(max(0, int(written_samples))),
        int(max(0, int(written_chunks))),
        int(max(0, int(total_samples))),
        int(max(0, int(speech_end_samples))),
        int(max(1, int(sample_capacity))),
        int(max(1, int(chunk_capacity))),
        int(max(1, int(sample_rate))),
        1 if bool(done) else 0,
    )


def live_audio_shm_read_header(shm_buf: memoryview) -> dict[str, Any] | None:
    if len(shm_buf) < int(LIVE_AUDIO_SHM_HEADER_BYTES):
        return None
    try:
        (
            magic,
            version,
            header_bytes,
            written_samples,
            written_chunks,
            total_samples,
            speech_end_samples,
            sample_capacity,
            chunk_capacity,
            sample_rate,
            done_flag,
        ) = LIVE_AUDIO_SHM_HEADER_STRUCT.unpack_from(shm_buf, 0)
    except Exception:
        return None
    if bytes(magic) != bytes(LIVE_AUDIO_SHM_MAGIC):
        return None
    if int(version) != int(LIVE_AUDIO_SHM_VERSION):
        return None
    if int(header_bytes) < int(LIVE_AUDIO_SHM_HEADER_BYTES):
        return None
    return {
        "written_samples": int(max(0, int(written_samples))),
        "written_chunks": int(max(0, int(written_chunks))),
        "total_samples": int(max(0, int(total_samples))),
        "speech_end_samples": int(max(0, int(speech_end_samples))),
        "sample_capacity": int(max(1, int(sample_capacity))),
        "chunk_capacity": int(max(1, int(chunk_capacity))),
        "sample_rate": int(max(1, int(sample_rate))),
        "done": bool(done_flag),
        "transport": "shm_ring",
        "header_bytes": int(max(0, int(header_bytes))),
    }


def live_audio_shm_write_chunk_meta(
    shm_buf: memoryview,
    *,
    chunk_capacity: int,
    chunk_idx: int,
    start_sample: int,
    sample_count: int,
    source_samples: int,
    audible: bool,
    kind: str,
    turn_done: bool,
) -> None:
    chunk_cap = int(max(1, int(chunk_capacity)))
    chunk_idx_i = int(max(1, int(chunk_idx)))
    kind_norm = str(kind or "speech").strip().lower()
    if kind_norm not in _KIND_TO_CODE:
        kind_norm = "speech"
    slot = int((chunk_idx_i - 1) % chunk_cap)
    offset = int(LIVE_AUDIO_SHM_HEADER_BYTES + (slot * LIVE_AUDIO_CHUNK_META_BYTES))
    LIVE_AUDIO_CHUNK_META_STRUCT.pack_into(
        shm_buf,
        offset,
        int(chunk_idx_i),
        int(max(0, int(start_sample))),
        int(max(0, int(sample_count))),
        int(max(0, int(source_samples))),
        1 if bool(audible) else 0,
        int(_KIND_TO_CODE[kind_norm]),
        1 if bool(turn_done) else 0,
    )


def live_audio_shm_read_chunk_meta(
    shm_buf: memoryview,
    *,
    chunk_capacity: int,
    chunk_idx: int,
) -> dict[str, Any] | None:
    chunk_cap = int(max(1, int(chunk_capacity)))
    chunk_idx_i = int(max(1, int(chunk_idx)))
    slot = int((chunk_idx_i - 1) % chunk_cap)
    offset = int(LIVE_AUDIO_SHM_HEADER_BYTES + (slot * LIVE_AUDIO_CHUNK_META_BYTES))
    try:
        (
            stored_chunk_idx,
            start_sample,
            sample_count,
            source_samples,
            audible_flag,
            kind_code,
            turn_done_flag,
        ) = LIVE_AUDIO_CHUNK_META_STRUCT.unpack_from(shm_buf, offset)
    except Exception:
        return None
    if int(stored_chunk_idx) != int(chunk_idx_i):
        return None
    return {
        "chunk_idx": int(max(1, int(stored_chunk_idx))),
        "start_sample": int(max(0, int(start_sample))),
        "sample_count": int(max(0, int(sample_count))),
        "source_samples": int(max(0, int(source_samples))),
        "audible": bool(audible_flag),
        "kind": str(_CODE_TO_KIND.get(int(kind_code), "speech")),
        "turn_done": bool(turn_done_flag),
    }


def live_audio_shm_write_pcm16le(
    shm_buf: memoryview,
    *,
    chunk_capacity: int,
    sample_capacity: int,
    start_sample: int,
    pcm_bytes: bytes,
) -> int:
    sample_cap = int(max(1, int(sample_capacity)))
    start = int(max(0, int(start_sample)))
    pcm = bytes(pcm_bytes or b"")
    if (len(pcm) % 2) != 0:
        pcm = pcm[: len(pcm) - 1]
    sample_count = int(len(pcm) // 2)
    if sample_count <= 0:
        return 0
    sample_region = live_audio_shm_sample_region(shm_buf, chunk_capacity=int(chunk_capacity))
    start_byte = int((start % sample_cap) * 2)
    total_bytes = int(sample_count * 2)
    first_nbytes = int(min(total_bytes, max(0, (sample_cap * 2) - start_byte)))
    if first_nbytes > 0:
        sample_region[start_byte : start_byte + first_nbytes] = pcm[:first_nbytes]
    remain = int(total_bytes - first_nbytes)
    if remain > 0:
        sample_region[0:remain] = pcm[first_nbytes:]
    return int(sample_count)


def live_audio_shm_read_pcm16le(
    shm_buf: memoryview,
    *,
    chunk_capacity: int,
    sample_capacity: int,
    written_samples: int,
    start_sample: int,
    sample_count: int,
) -> bytes | None:
    sample_cap = int(max(1, int(sample_capacity)))
    written = int(max(0, int(written_samples)))
    start = int(max(0, int(start_sample)))
    count = int(max(0, int(sample_count)))
    if count <= 0:
        return b""
    if int(start + count) > int(written):
        return None
    if int(start) < int(max(0, int(written - sample_cap))):
        return None
    sample_region = live_audio_shm_sample_region(shm_buf, chunk_capacity=int(chunk_capacity))
    start_byte = int((start % sample_cap) * 2)
    total_bytes = int(count * 2)
    first_nbytes = int(min(total_bytes, max(0, (sample_cap * 2) - start_byte)))
    out = bytearray()
    if first_nbytes > 0:
        out.extend(sample_region[start_byte : start_byte + first_nbytes])
    remain = int(total_bytes - first_nbytes)
    if remain > 0:
        out.extend(sample_region[0:remain])
    return bytes(out)


@dataclass
class LiveAudioShmWriter:
    queue_dir: str
    sample_rate: int
    sample_capacity: int
    chunk_capacity: int
    shm: shared_memory.SharedMemory
    written_samples: int = 0
    written_chunks: int = 0
    total_samples: int = 0
    speech_end_samples: int = 0
    done: bool = False

    @classmethod
    def create(
        cls,
        *,
        queue_dir: str,
        sample_rate: int,
        sample_capacity: int,
        chunk_capacity: int,
    ) -> "LiveAudioShmWriter":
        qdir = os.path.abspath(str(queue_dir or "").strip())
        os.makedirs(qdir, exist_ok=True)
        shm_name = f"avalife_live_audio_{os.getpid()}_{int(time.time() * 1000.0)}"
        shm = shared_memory.SharedMemory(
            name=str(shm_name),
            create=True,
            size=int(
                live_audio_shm_total_bytes(
                    sample_capacity=int(sample_capacity),
                    chunk_capacity=int(chunk_capacity),
                )
            ),
        )
        writer = cls(
            queue_dir=str(qdir),
            sample_rate=int(max(1, int(sample_rate))),
            sample_capacity=int(max(1, int(sample_capacity))),
            chunk_capacity=int(max(1, int(chunk_capacity))),
            shm=shm,
        )
        live_audio_shm_write_header(
            writer.shm.buf,
            written_samples=0,
            written_chunks=0,
            total_samples=0,
            speech_end_samples=0,
            sample_capacity=int(writer.sample_capacity),
            chunk_capacity=int(writer.chunk_capacity),
            sample_rate=int(writer.sample_rate),
            done=False,
        )
        meta = {
            "transport": "shm_ring",
            "shm_name": str(writer.shm.name),
            "sample_capacity": int(writer.sample_capacity),
            "chunk_capacity": int(writer.chunk_capacity),
            "sample_rate": int(writer.sample_rate),
            "header_bytes": int(LIVE_AUDIO_SHM_HEADER_BYTES),
            "chunk_meta_bytes": int(LIVE_AUDIO_CHUNK_META_BYTES),
            "ts_ms": int(time.time() * 1000.0),
        }
        tmp = os.path.join(qdir, "audio_shm.json.tmp")
        final = os.path.join(qdir, "audio_shm.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)
        os.replace(tmp, final)
        return writer

    def write_chunk(
        self,
        *,
        chunk_idx: int,
        pcm_bytes: bytes,
        kind: str,
        audible: bool,
        source_samples: int | None = None,
        turn_done: bool = False,
    ) -> int:
        pcm = bytes(pcm_bytes or b"")
        if (len(pcm) % 2) != 0:
            pcm = pcm[: len(pcm) - 1]
        sample_count = int(len(pcm) // 2)
        if sample_count <= 0:
            return 0
        start_sample = int(self.written_samples)
        written = live_audio_shm_write_pcm16le(
            self.shm.buf,
            chunk_capacity=int(self.chunk_capacity),
            sample_capacity=int(self.sample_capacity),
            start_sample=int(start_sample),
            pcm_bytes=pcm,
        )
        if written <= 0:
            return 0
        live_audio_shm_write_chunk_meta(
            self.shm.buf,
            chunk_capacity=int(self.chunk_capacity),
            chunk_idx=int(chunk_idx),
            start_sample=int(start_sample),
            sample_count=int(written),
            source_samples=(
                int(max(0, int(source_samples)))
                if isinstance(source_samples, (int, float))
                else int(written)
            ),
            audible=bool(audible),
            kind=str(kind or "speech"),
            turn_done=bool(turn_done),
        )
        self.written_samples = int(start_sample + written)
        self.written_chunks = int(max(int(self.written_chunks), int(chunk_idx)))
        self.total_samples = int(self.written_samples)
        live_audio_shm_write_header(
            self.shm.buf,
            written_samples=int(self.written_samples),
            written_chunks=int(self.written_chunks),
            total_samples=int(self.total_samples),
            speech_end_samples=int(self.speech_end_samples),
            sample_capacity=int(self.sample_capacity),
            chunk_capacity=int(self.chunk_capacity),
            sample_rate=int(self.sample_rate),
            done=bool(self.done),
        )
        return int(written)

    def mark_done(
        self,
        *,
        total_samples: int | None = None,
        speech_end_samples: int | None = None,
    ) -> None:
        if isinstance(total_samples, (int, float)):
            self.total_samples = int(max(0, int(total_samples)))
        if isinstance(speech_end_samples, (int, float)):
            self.speech_end_samples = int(max(0, int(speech_end_samples)))
        self.done = True
        live_audio_shm_write_header(
            self.shm.buf,
            written_samples=int(self.written_samples),
            written_chunks=int(self.written_chunks),
            total_samples=int(self.total_samples),
            speech_end_samples=int(self.speech_end_samples),
            sample_capacity=int(self.sample_capacity),
            chunk_capacity=int(self.chunk_capacity),
            sample_rate=int(self.sample_rate),
            done=True,
        )

    def close(self) -> None:
        try:
            self.shm.close()
        except Exception:
            pass
