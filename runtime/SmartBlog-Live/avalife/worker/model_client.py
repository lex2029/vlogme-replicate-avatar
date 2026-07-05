from __future__ import annotations

from .common import *
from avalife.model.config import model_runtime_socket_path
from avalife.model.protocol import (
    CancelMediaRequest,
    CancelMediaResponse,
    CancelInferRequest,
    CancelInferResponse,
    cancel_media_request,
    InferRequest,
    InferResponse,
    MediaProcessRequest,
    MediaProcessResponse,
    cancel_infer_request,
    infer_request,
    media_process_request,
    ping_request,
)


class ModelRuntimeClient:
    def __init__(self, *, socket_path: str | None = None) -> None:
        self.socket_path = os.path.abspath(str(socket_path or model_runtime_socket_path()).strip())

    async def wait_ready(self, *, timeout_sec: float = 900.0) -> None:
        deadline = float(time.perf_counter() + max(1.0, float(timeout_sec)))
        last_error = "not started"
        while time.perf_counter() < deadline:
            try:
                resp = await self._request(ping_request())
                if bool(resp.get("ok")):
                    return
                last_error = str(resp.get("error") or "ping failed")
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
            await asyncio.sleep(0.25)
        raise TimeoutError(f"model runtime not ready: {last_error}")

    async def infer(self, *, req: InferRequest) -> InferResponse:
        payload = await self._request(infer_request(req))
        return InferResponse.from_payload(payload)

    async def media_process(self, *, req: MediaProcessRequest) -> MediaProcessResponse:
        payload = await self._request(media_process_request(req))
        return MediaProcessResponse.from_payload(payload)

    async def cancel_active_infer(self, *, reason: str = "") -> CancelInferResponse:
        payload = await self._request(cancel_infer_request(CancelInferRequest(reason=str(reason or ""))))
        return CancelInferResponse.from_payload(payload)

    async def cancel_active_media(self, *, reason: str = "") -> CancelMediaResponse:
        payload = await self._request(cancel_media_request(CancelMediaRequest(reason=str(reason or ""))))
        return CancelMediaResponse.from_payload(payload)

    async def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        try:
            writer.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            await writer.drain()
            line = await reader.readline()
            if not line:
                raise RuntimeError("empty response from model runtime")
            resp = json.loads(line.decode("utf-8"))
            if not isinstance(resp, dict):
                raise RuntimeError("invalid model runtime response")
            return dict(resp)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
