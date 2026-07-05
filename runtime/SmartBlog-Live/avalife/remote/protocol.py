from __future__ import annotations

import asyncio
import json
import time
from typing import Any

MAX_HEADER_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 512 * 1024 * 1024


async def read_message(reader: asyncio.StreamReader) -> tuple[dict[str, Any], bytes]:
    line = await reader.readline()
    if not line:
        raise EOFError("remote stream closed")
    if len(line) > MAX_HEADER_BYTES:
        raise ValueError("remote stream header too large")
    header = json.loads(line.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("remote stream header must be object")
    payload_len = int(header.get("payload_len") or 0)
    if payload_len < 0 or payload_len > MAX_PAYLOAD_BYTES:
        raise ValueError(f"invalid remote stream payload_len={payload_len}")
    payload = await reader.readexactly(payload_len) if payload_len else b""
    return dict(header), payload


async def read_message_timed(reader: asyncio.StreamReader) -> tuple[dict[str, Any], bytes, dict[str, float]]:
    started = time.perf_counter()
    line = await reader.readline()
    header_read_sec = time.perf_counter() - started
    if not line:
        raise EOFError("remote stream closed")
    if len(line) > MAX_HEADER_BYTES:
        raise ValueError("remote stream header too large")
    header = json.loads(line.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("remote stream header must be object")
    payload_len = int(header.get("payload_len") or 0)
    if payload_len < 0 or payload_len > MAX_PAYLOAD_BYTES:
        raise ValueError(f"invalid remote stream payload_len={payload_len}")
    payload_started = time.perf_counter()
    payload = await reader.readexactly(payload_len) if payload_len else b""
    payload_read_sec = time.perf_counter() - payload_started
    return dict(header), payload, {
        "header_read_sec": float(header_read_sec),
        "payload_read_sec": float(payload_read_sec),
        "total_read_sec": float(time.perf_counter() - started),
    }


async def write_message(
    writer: asyncio.StreamWriter,
    message_type: str,
    payload: bytes | bytearray | memoryview = b"",
    **fields: Any,
) -> None:
    payload_b = bytes(payload)
    header = {"type": str(message_type), "payload_len": len(payload_b)}
    header.update(fields)
    writer.write((json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    if payload_b:
        writer.write(payload_b)
    await writer.drain()
