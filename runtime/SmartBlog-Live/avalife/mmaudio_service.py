from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import shutil
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from avalife.core.upload_retry import put_file_to_signed_url
from scipy.io import wavfile

from mmaudio.eval_utils import ModelConfig, all_model_cfg, generate, load_video
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils


LOG = logging.getLogger("smartblog-mmaudio-service")
_MMAUDIO_SERVICE_SEMAPHORE: threading.Semaphore | None = None


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default)) or str(default)).strip())
    except Exception:
        return int(default)


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(int(status))
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _mmaudio_service_semaphore() -> threading.Semaphore:
    global _MMAUDIO_SERVICE_SEMAPHORE
    if _MMAUDIO_SERVICE_SEMAPHORE is None:
        max_concurrent = max(1, _env_int("SMARTBLOG_MMAUDIO_SERVICE_MAX_CONCURRENT", 1))
        _MMAUDIO_SERVICE_SEMAPHORE = threading.Semaphore(max_concurrent)
    return _MMAUDIO_SERVICE_SEMAPHORE


def _request_work_dir(prefix: str) -> Path:
    root = Path(os.getenv("SMARTBLOG_MMAUDIO_SERVICE_REQUEST_DIR", "/tmp/smartblog-mmaudio-service-requests"))
    path = root / f"{prefix}_{os.getpid()}_{int(time.time() * 1000)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_suffix(name: str, default: str) -> str:
    suffix = Path(str(name or "")).suffix.lower()
    if not suffix or len(suffix) > 12:
        suffix = str(default)
    return suffix


def _write_base64_file(value: Any, *, out_dir: Path, prefix: str, default_ext: str) -> str:
    filename = ""
    content = value
    if isinstance(value, dict):
        filename = str(value.get("filename") or value.get("name") or "")
        content = value.get("data") or value.get("base64") or value.get("content") or ""
    text = str(content or "").strip()
    if "," in text[:128]:
        text = text.split(",", 1)[1]
    path = out_dir / f"{prefix}_{abs(hash(text[:512])) % 1000000}{_safe_suffix(filename, default_ext)}"
    with open(path, "wb") as f:
        f.write(base64.b64decode(text))
    return str(path)


def _download_url_file(url: str, *, out_dir: Path, prefix: str, default_ext: str) -> str:
    raw = str(url or "").strip()
    suffix = _safe_suffix(urllib.parse.urlparse(raw).path, default_ext)
    path = out_dir / f"{prefix}_{abs(hash(raw)) % 1000000}{suffix}"
    with urllib.request.urlopen(raw, timeout=600) as resp, open(path, "wb") as f:
        shutil.copyfileobj(resp, f)
    return str(path)


def _materialize_mmaudio_request_inputs(request: dict[str, Any]) -> dict[str, Any]:
    out = dict(request or {})
    if out.get("video_base64"):
        work_dir = _request_work_dir("input")
        out["video_path"] = _write_base64_file(out.get("video_base64"), out_dir=work_dir, prefix="video", default_ext=".mp4")
    elif out.get("video_url"):
        work_dir = _request_work_dir("input")
        out["video_path"] = _download_url_file(str(out.get("video_url")), out_dir=work_dir, prefix="video", default_ext=".mp4")
    return out


def _upload_file_to_signed_url(*, signed_url: str, path: str, content_type: str) -> None:
    url = str(signed_url or "").strip()
    if not url:
        return
    put_file_to_signed_url(
        signed_url=str(url),
        path=str(path),
        content_type=str(content_type or "application/octet-stream"),
        connect_timeout=20.0,
        read_timeout=1800.0,
        env_prefix="SMARTBLOG_MMAUDIO_UPLOAD",
        log_prefix="mmaudio-signed-upload",
    )


def _publish_output_if_requested(response: dict[str, Any], *, output_path: str, request: dict[str, Any], content_type: str) -> dict[str, Any]:
    out = dict(response or {})
    upload_url = str(request.get("output_upload_url") or request.get("upload_url") or "").strip()
    if upload_url:
        _upload_file_to_signed_url(signed_url=upload_url, path=str(output_path), content_type=str(content_type))
        out["uploaded"] = True
        if request.get("output_storage_path"):
            out["output_storage_path"] = str(request.get("output_storage_path"))
        if request.get("output_public_url") or request.get("output_url"):
            out["output_url"] = str(request.get("output_public_url") or request.get("output_url"))
    if _env_flag("SMARTBLOG_MMAUDIO_SERVICE_RETURN_BASE64", "0") or bool(request.get("return_base64")):
        with open(str(output_path), "rb") as f:
            out["output_base64"] = base64.b64encode(f.read()).decode("ascii")
    return out


def _sync_cuda_if_needed(device: str) -> None:
    if torch.cuda.is_available() and str(device or "").startswith("cuda"):
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


class MMAudioResidentPipeline:
    def __init__(
        self,
        *,
        variant: str,
        device: str | None = None,
        dtype: str = "bf16",
        num_steps: int = 25,
        cfg_strength: float = 4.5,
    ) -> None:
        if str(variant) not in all_model_cfg:
            raise ValueError(f"Unknown MMAudio variant: {variant}")
        self.variant = str(variant)
        self.model_cfg: ModelConfig = all_model_cfg[self.variant]
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype_s = str(dtype or "bf16").strip().lower()
        self.dtype = torch.float32 if dtype_s in {"fp32", "float32", "full", "full_precision"} else torch.bfloat16
        if dtype_s in {"fp16", "float16", "half"}:
            self.dtype = torch.float16
        self.default_num_steps = int(max(1, int(num_steps or 25)))
        self.default_cfg_strength = float(max(0.0, float(cfg_strength or 4.5)))
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.ready_at = 0.0
        self.net: MMAudio
        self.feature_utils: FeaturesUtils
        self._load()
        self.ready_at = time.time()

    def _load(self) -> None:
        LOG.warning(
            "loading MMAudio variant=%s dtype=%s device=%s",
            self.variant,
            str(self.dtype),
            self.device,
        )
        self.model_cfg.download_if_needed()
        net: MMAudio = get_my_mmaudio(self.model_cfg.model_name).to(self.device, self.dtype).eval()
        net.load_weights(torch.load(self.model_cfg.model_path, map_location=self.device, weights_only=True))
        feature_utils = FeaturesUtils(
            tod_vae_ckpt=self.model_cfg.vae_path,
            synchformer_ckpt=self.model_cfg.synchformer_ckpt,
            enable_conditions=True,
            mode=self.model_cfg.mode,
            bigvgan_vocoder_ckpt=self.model_cfg.bigvgan_16k_path,
            need_vae_encoder=False,
        )
        feature_utils = feature_utils.to(self.device, self.dtype).eval()
        self.net = net
        self.feature_utils = feature_utils
        LOG.warning("MMAudio pipeline loaded in %.2fs", float(time.time() - self.started_at))

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        request = _materialize_mmaudio_request_inputs(dict(request or {}))
        video_path = str(request.get("video_path") or request.get("video") or "").strip()
        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"MMAudio requires existing video_path, got: {video_path or '-'}")
        prompt = str(request.get("prompt") or "").strip()
        negative_prompt = str(request.get("negative_prompt") or "").strip()
        duration = float(max(0.1, min(60.0, float(request.get("duration") or request.get("duration_sec") or 8.0))))
        seed = int(request.get("seed") if request.get("seed") is not None else int(os.getenv("SMARTBLOG_MMAUDIO_SEED", "42")))
        num_steps = int(max(1, int(request.get("num_steps") or self.default_num_steps)))
        cfg_strength = float(max(0.0, float(request.get("cfg_strength") or self.default_cfg_strength)))
        output_raw = str(request.get("output_path") or "").strip()
        if output_raw:
            output_path = Path(output_raw).expanduser()
        else:
            output_path = Path("outputs/mmaudio") / f"{Path(video_path).stem}.wav"
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with self.lock:
            timings: dict[str, float] = {}
            t0 = time.perf_counter()

            def mark(name: str, started: float) -> None:
                timings[name] = timings.get(name, 0.0) + float(time.perf_counter() - started)

            LOG.warning(
                "generate MMAudio variant=%s video=%s duration=%.3fs steps=%d cfg=%.2f seed=%d prompt_chars=%d negative_chars=%d",
                self.variant,
                os.path.basename(str(video_path)),
                float(duration),
                int(num_steps),
                float(cfg_strength),
                int(seed),
                int(len(prompt)),
                int(len(negative_prompt)),
            )
            prep_t0 = time.perf_counter()
            video_info = load_video(Path(video_path), duration)
            clip_frames = video_info.clip_frames.unsqueeze(0)
            sync_frames = video_info.sync_frames.unsqueeze(0)
            seq_cfg = self.model_cfg.seq_cfg
            seq_cfg.duration = float(video_info.duration_sec)
            self.net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
            rng = torch.Generator(device=self.device)
            rng.manual_seed(int(seed))
            fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=int(num_steps))
            mark("prepare", prep_t0)

            _sync_cuda_if_needed(self.device)
            infer_t0 = time.perf_counter()
            with torch.no_grad():
                audios = generate(
                    clip_frames,
                    sync_frames,
                    [prompt],
                    negative_text=[negative_prompt],
                    feature_utils=self.feature_utils,
                    net=self.net,
                    fm=fm,
                    rng=rng,
                    cfg_strength=float(cfg_strength),
                )
            _sync_cuda_if_needed(self.device)
            mark("inference", infer_t0)

            write_t0 = time.perf_counter()
            audio = audios.float().cpu()[0].numpy()
            if audio.ndim == 2:
                audio = audio.T
            wavfile.write(str(output_path), int(seq_cfg.sampling_rate), audio)
            mark("file_write", write_t0)
            elapsed = float(time.perf_counter() - t0)
            LOG.warning(
                "MMAudio generate done in %.2fs prepare=%.2fs inference=%.2fs file_write=%.2fs output=%s",
                elapsed,
                float(timings.get("prepare", 0.0)),
                float(timings.get("inference", 0.0)),
                float(timings.get("file_write", 0.0)),
                str(output_path),
            )
            return _publish_output_if_requested(
                {
                    "ok": True,
                    "output_path": str(output_path),
                    "duration_sec": float(video_info.duration_sec),
                    "sample_rate": int(seq_cfg.sampling_rate),
                    "elapsed_sec": elapsed,
                    "timings_sec": dict(timings),
                },
                output_path=str(output_path),
                request=request,
                content_type="audio/wav",
            )


class MMAudioHandler(BaseHTTPRequestHandler):
    server_version = "SmartBlogMMAudioService/1.0"

    @property
    def resident(self) -> MMAudioResidentPipeline:
        return getattr(self.server, "resident")  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def _run_generate_queued(self, payload: dict[str, Any]) -> dict[str, Any]:
        sem = _mmaudio_service_semaphore()
        queued_at = time.perf_counter()
        acquired_immediately = sem.acquire(blocking=False)
        if not acquired_immediately:
            LOG.warning("MMAudio service queued: another GPU request is running")
            sem.acquire()
        queue_sec = float(time.perf_counter() - queued_at)
        try:
            result = self.resident.generate(payload)
            if isinstance(result, dict):
                result.setdefault("queue_sec", queue_sec)
            return result
        finally:
            sem.release()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"", "/health"}:
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "ready": True,
                    "pid": os.getpid(),
                    "device": self.resident.device,
                    "variant": self.resident.variant,
                    "ready_at": self.resident.ready_at,
                },
            )
            return
        _json_response(self, 404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8") or "{}")
            if self.path.rstrip("/") == "/generate":
                _json_response(self, 200, self._run_generate_queued(payload))
                return
            _json_response(self, 404, {"ok": False, "error": "not_found"})
        except Exception as e:
            LOG.exception("request failed")
            _json_response(self, 500, {"ok": False, "error": str(e)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("SMARTBLOG_MMAUDIO_SERVICE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SMARTBLOG_MMAUDIO_SERVICE_PORT", "8799")))
    parser.add_argument("--variant", default=os.getenv("SMARTBLOG_MMAUDIO_VARIANT", "large_44k_v2"))
    parser.add_argument("--device", default=os.getenv("SMARTBLOG_MMAUDIO_SERVICE_DEVICE", "cuda"))
    parser.add_argument("--dtype", default=os.getenv("SMARTBLOG_MMAUDIO_DTYPE", "bf16"))
    parser.add_argument("--num-steps", type=int, default=int(os.getenv("SMARTBLOG_MMAUDIO_NUM_STEPS", "25")))
    parser.add_argument("--cfg-strength", type=float, default=float(os.getenv("SMARTBLOG_MMAUDIO_CFG_STRENGTH", "4.5")))
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(os.getenv("SMARTBLOG_MMAUDIO_SERVICE_LOG_LEVEL", "INFO")).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    resident = MMAudioResidentPipeline(
        variant=str(args.variant),
        device=str(args.device),
        dtype=str(args.dtype),
        num_steps=int(args.num_steps),
        cfg_strength=float(args.cfg_strength),
    )
    server = ThreadingHTTPServer((str(args.host), int(args.port)), MMAudioHandler)
    setattr(server, "resident", resident)
    LOG.warning("MMAudio service listening on %s:%d pid=%d", str(args.host), int(args.port), os.getpid())
    server.serve_forever()


if __name__ == "__main__":
    main()
