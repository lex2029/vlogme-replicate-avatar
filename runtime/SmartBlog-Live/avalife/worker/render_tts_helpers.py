from __future__ import annotations

import base64
import io
import math
import re
import urllib.parse
import wave

import aiohttp

from .common import *


_RENDER_TTS_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u27BF"
    "\u200D"
    "\uFE0F"
    "\u20E3"
    "]+",
    flags=re.UNICODE,
)
_RENDER_TTS_END_MARKER_RE = re.compile(r"\s*!END!\s*[\.\?!…。\"'”’»\]\)]*\s*$", flags=re.IGNORECASE)


class SmartBlogRenderTTSMixin:
    _TTS_WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)

    async def _get_render_eleven_http_session(self) -> aiohttp.ClientSession:
        session = getattr(self, "_render_eleven_http_session", None)
        if isinstance(session, aiohttp.ClientSession) and (not session.closed):
            return session

        async with self._render_tts_clients_lock:
            session = getattr(self, "_render_eleven_http_session", None)
            if isinstance(session, aiohttp.ClientSession) and (not session.closed):
                return session

            connector = aiohttp.TCPConnector(
                limit=32,
                enable_cleanup_closed=True,
                ttl_dns_cache=300,
            )
            connect_timeout = max(
                1.0,
                min(60.0, float(_safe_float_env("WORKER_ELEVEN_HTTP_CONNECT_TIMEOUT_SEC", 15.0))),
            )
            sock_read_raw = float(_safe_float_env("WORKER_ELEVEN_HTTP_SOCK_READ_TIMEOUT_SEC", 20.0))
            sock_read_timeout = None
            if sock_read_raw > 0.0:
                sock_read_timeout = max(5.0, min(180.0, float(sock_read_raw)))
            timeout = aiohttp.ClientTimeout(
                total=None,
                connect=float(connect_timeout),
                sock_connect=float(connect_timeout),
                sock_read=sock_read_timeout,
            )
            session = aiohttp.ClientSession(connector=connector, timeout=timeout)
            self._render_eleven_http_session = session
            logging.info("Render TTS transport: created shared Eleven HTTP session")
            return session

    async def _close_render_tts_clients(self) -> None:
        async with self._render_tts_clients_lock:
            eleven_session = getattr(self, "_render_eleven_http_session", None)
            self._render_eleven_http_session = None
        if isinstance(eleven_session, aiohttp.ClientSession) and (not eleven_session.closed):
            try:
                await eleven_session.close()
            except Exception as e:
                logging.warning("Render TTS transport: Eleven session close failed: %s", e)

    def _runtime_eleven_api_key(self) -> str:
        return str(_worker_secret_env(self, "ELEVENLABS_API_KEY") or "").strip()

    @staticmethod
    def _render_eleven_v3_seed() -> int | None:
        raw = str(os.getenv("WORKER_ELEVEN_V3_SEED", "") or "").strip()
        if not raw:
            raw = str(os.getenv("BASE_SEED", "420") or "420").strip()
        try:
            value = int(raw)
        except Exception:
            return None
        if int(value) < 0:
            return None
        return int(max(0, min(4294967295, int(value))))

    @staticmethod
    def _render_has_terminal_punct(text: str) -> bool:
        s = str(text or "")
        if not s.strip():
            return False
        closers = set("\"'”’»)]}")
        i = len(s) - 1
        while i >= 0 and s[i].isspace():
            i -= 1
        while i >= 0 and s[i] in closers:
            i -= 1
        return i >= 0 and s[i] in ".!?"

    @classmethod
    def _ensure_render_terminal_punct(cls, text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        if cls._render_has_terminal_punct(s):
            return s
        return f"{s}."

    @staticmethod
    def _strip_render_emoji_chars(text: str) -> str:
        s = str(text or "")
        if not s:
            return ""
        s = _RENDER_TTS_EMOJI_RE.sub(" ", s)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r" *\n *", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        s = re.sub(r"\s+([,.;:!?])", r"\1", s)
        return s.strip()

    @classmethod
    def _sanitize_render_tts_delta(cls, text: str) -> str:
        s = str(text or "")
        if not s:
            return ""
        out_chars: list[str] = []
        for ch in s:
            if ch.isalnum() or ch.isspace():
                out_chars.append(ch)
                continue
            if ch in ".,?!,":
                out_chars.append(ch)
                continue
            if ch in "[]":
                out_chars.append(ch)
                continue
            out_chars.append(" ")
        out = "".join(out_chars)
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r"\s+([,.?!\]])", r"\1", out)
        out = re.sub(r"([\[])\s+", r"\1", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
        return out

    @classmethod
    def _sanitize_render_tts_text(cls, text: str) -> str:
        s = cls._strip_render_emoji_chars(text)
        if not s:
            return ""
        out = cls._sanitize_render_tts_delta(s)
        out = re.sub(r"([,.?!])(?:\s*[,.?!])+", r"\1", out)
        return out.strip()

    @classmethod
    def _split_render_end_marker(cls, text: str) -> tuple[str, bool]:
        s = str(text or "")
        if not s:
            return "", False
        m = _RENDER_TTS_END_MARKER_RE.search(s)
        if not m:
            return s, False
        return s[: m.start()].rstrip(), True

    @classmethod
    def _sanitize_render_text(cls, text: str) -> str:
        stripped, _ = cls._split_render_end_marker(text)
        s = cls._sanitize_render_tts_text(stripped)
        if not s:
            return ""
        return cls._ensure_render_terminal_punct(s)

    @classmethod
    def _sanitize_render_description_text(cls, text: str) -> str:
        s, _ = cls._split_render_end_marker(text)
        if not s:
            return ""
        s = re.sub(r"\[[^\]\n]{0,120}\]", " ", s)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r" *\n *", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return cls._sanitize_render_text(s)

    @classmethod
    def _render_tts_word_count(cls, text: str) -> int:
        return int(len(cls._TTS_WORD_RE.findall(str(text or ""))))

    @staticmethod
    def _render_eleven_http_segment_timeout_sec(*, chars: int, words: int) -> float:
        min_timeout = float(
            max(
                5.0,
                min(120.0, _safe_float_env("WORKER_ELEVEN_HTTP_SEGMENT_MIN_TIMEOUT_SEC", 12.0)),
            )
        )
        max_timeout = float(
            max(
                min_timeout,
                min(300.0, _safe_float_env("WORKER_ELEVEN_HTTP_SEGMENT_MAX_TIMEOUT_SEC", 45.0)),
            )
        )
        base = float(max(1.0, min(60.0, _safe_float_env("WORKER_ELEVEN_HTTP_SEGMENT_TIMEOUT_BASE_SEC", 8.0))))
        per_word = float(
            max(0.0, min(5.0, _safe_float_env("WORKER_ELEVEN_HTTP_SEGMENT_TIMEOUT_PER_WORD_SEC", 0.6)))
        )
        per_char = float(
            max(0.0, min(1.0, _safe_float_env("WORKER_ELEVEN_HTTP_SEGMENT_TIMEOUT_PER_CHAR_SEC", 0.02)))
        )
        raw = float(base) + float(max(0, int(words))) * float(per_word) + float(max(0, int(chars))) * float(per_char)
        return float(max(min_timeout, min(max_timeout, raw)))

    @staticmethod
    def _render_eleven_forced_alignment_timeout_sec(*, chars: int, samples: int, sample_rate: int) -> float:
        env_timeout = float(_safe_float_env("WORKER_ELEVEN_FORCED_ALIGNMENT_TIMEOUT_SEC", 0.0))
        if env_timeout > 0.0:
            return float(max(2.0, min(120.0, env_timeout)))
        duration_sec = float(max(0, int(samples))) / float(max(1, int(sample_rate)))
        return float(max(6.0, min(45.0, 4.0 + duration_sec * 0.75 + float(max(0, int(chars))) * 0.025)))

    @staticmethod
    def _render_pcm16le_wav_bytes(pcm: bytes, *, sample_rate: int) -> bytes:
        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(bytes(pcm or b""))
        return bytes(bio.getvalue())

    @staticmethod
    def _render_forced_alignment_to_character_alignment(data: dict[str, Any]) -> dict[str, Any] | None:
        chars_raw = data.get("characters") if isinstance(data, dict) else None
        if not isinstance(chars_raw, list) or not chars_raw:
            return None

        chars: list[str] = []
        starts: list[float] = []
        ends: list[float] = []
        for item in chars_raw:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("character") or item.get("char") or "")
                start_raw = item.get("start", item.get("start_time", item.get("start_sec")))
                end_raw = item.get("end", item.get("end_time", item.get("end_sec")))
            else:
                text = str(item or "")
                start_raw = None
                end_raw = None
            if not text:
                continue
            try:
                start = float(start_raw)
                end = float(end_raw)
            except Exception:
                continue
            if not math.isfinite(start) or not math.isfinite(end):
                continue
            end = max(float(start), float(end))
            pieces = list(text)
            if len(pieces) == 1:
                chars.append(str(pieces[0]))
                starts.append(float(start))
                ends.append(float(end))
                continue
            span = max(0.0, float(end) - float(start))
            for idx, ch in enumerate(pieces):
                ch_start = float(start) + (span * float(idx) / float(len(pieces)))
                ch_end = float(start) + (span * float(idx + 1) / float(len(pieces)))
                chars.append(str(ch))
                starts.append(float(ch_start))
                ends.append(float(max(ch_start, ch_end)))
        if not chars:
            return None
        return {
            "characters": list(chars),
            "character_start_times_seconds": list(starts),
            "character_end_times_seconds": list(ends),
        }

    async def _render_eleven_forced_align_subtitles(
        self,
        *,
        session: aiohttp.ClientSession,
        api_key: str,
        pcm: bytes,
        sample_rate: int,
        text: str,
        trace: str,
        segment_index: int,
    ) -> dict[str, Any] | None:
        text_s = str(text or "").strip()
        pcm_b = bytes(pcm or b"")
        if not text_s or not pcm_b:
            return None
        if len(pcm_b) % 2:
            pcm_b = pcm_b[: len(pcm_b) - 1]
        if not pcm_b:
            return None
        timeout_sec = self._render_eleven_forced_alignment_timeout_sec(
            chars=int(len(text_s)),
            samples=int(len(pcm_b) // 2),
            sample_rate=int(sample_rate),
        )
        wav_bytes = self._render_pcm16le_wav_bytes(pcm_b, sample_rate=int(sample_rate))
        form = aiohttp.FormData()
        form.add_field(
            "file",
            wav_bytes,
            filename=f"smartblog_segment_{int(segment_index):04d}.wav",
            content_type="audio/wav",
        )
        form.add_field("text", text_s)
        headers = {
            "xi-api-key": str(api_key),
            "accept": "application/json",
        }
        t_align = time.perf_counter()
        async with session.post(
            "https://api.elevenlabs.io/v1/forced-alignment",
            headers=headers,
            data=form,
            timeout=aiohttp.ClientTimeout(total=float(timeout_sec)),
        ) as resp:
            if resp.status >= 400:
                body = await resp.text(errors="replace")
                raise RuntimeError(f"Eleven forced-alignment error: HTTP {resp.status}: {body[:300]}")
            data = await resp.json(content_type=None)
        if not isinstance(data, dict):
            raise RuntimeError("Eleven forced-alignment returned non-object JSON")
        alignment = self._render_forced_alignment_to_character_alignment(data)
        if not isinstance(alignment, dict) or not alignment:
            raise RuntimeError("Eleven forced-alignment returned no character timings")
        logging.warning(
            "Render TTS forced-alignment subtitles ok: trace=%s segment=%d chars=%d text_chars=%d audio_sec=%.3f loss=%s dt=%.3fs",
            str(trace),
            int(segment_index),
            int(len(alignment.get("characters") or [])),
            int(len(text_s)),
            float(len(pcm_b) // 2) / float(max(1, int(sample_rate))),
            str(data.get("loss", "-")),
            float(time.perf_counter() - t_align),
        )
        return dict(alignment)

    def _build_eleven_http_tts_request(
        self,
        *,
        text: str,
        api_key: str,
        voice_id: str,
        voice_settings: dict[str, Any],
        model_id: str,
        seed: int | None = None,
        previous_text: str = "",
        next_text: str = "",
        previous_request_ids: list[str] | None = None,
        with_timestamps: bool = False,
    ) -> tuple[str, dict[str, str], dict[str, Any], str]:
        output_format = str(os.getenv("WORKER_ELEVEN_WS_OUTPUT_FORMAT", "pcm_16000") or "pcm_16000").strip()
        if not str(output_format).startswith("pcm_"):
            raise RuntimeError(f"Eleven HTTP TTS requires pcm_* output format, got {output_format!r}")
        try:
            output_sample_rate = int(str(output_format).split("_", 1)[1])
        except Exception:
            output_sample_rate = int(self.sample_rate)
        if int(output_sample_rate) != int(self.sample_rate):
            raise RuntimeError(
                f"Eleven HTTP TTS output sample rate {output_sample_rate} does not match worker sample rate {self.sample_rate}"
            )

        query = urllib.parse.urlencode({"output_format": str(output_format)})
        voice_path = urllib.parse.quote(str(voice_id), safe="")
        endpoint = "with-timestamps" if bool(with_timestamps) else "stream"
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_path}/{endpoint}?{query}"
        payload: dict[str, Any] = {
            "text": str(text or ""),
            "model_id": str(model_id),
        }
        settings = {
            k: v
            for k, v in dict(voice_settings or {}).items()
            if k in {"stability", "similarity_boost", "style", "speed", "use_speaker_boost"} and v is not None
        }
        if str(model_id or "").strip().lower() == "eleven_v3":
            defaults: dict[str, tuple[str, Any, type]] = {
                "stability": ("WORKER_ELEVEN_V3_DEFAULT_STABILITY", 0.85, float),
                "similarity_boost": ("WORKER_ELEVEN_V3_DEFAULT_SIMILARITY_BOOST", 0.95, float),
                "style": ("WORKER_ELEVEN_V3_DEFAULT_STYLE", 0.0, float),
                "use_speaker_boost": ("WORKER_ELEVEN_V3_DEFAULT_USE_SPEAKER_BOOST", True, bool),
            }
            for key, (env_name, default_value, caster) in defaults.items():
                if key in settings:
                    continue
                raw = os.getenv(str(env_name))
                value: Any = default_value if raw is None or str(raw).strip() == "" else raw
                try:
                    if caster is bool:
                        settings[key] = bool(str(value).strip().lower() not in {"0", "false", "no", "off"})
                    elif caster is float:
                        settings[key] = float(value)
                    else:
                        settings[key] = value
                except Exception:
                    settings[key] = default_value
        if settings:
            payload["voice_settings"] = dict(settings)
        if seed is not None:
            payload["seed"] = int(max(0, min(4294967295, int(seed))))

        supports_context_hints = str(model_id or "").strip().lower() != "eleven_v3"
        if bool(supports_context_hints):
            request_ids = [
                str(item or "").strip()
                for item in (previous_request_ids or [])
                if str(item or "").strip()
            ][-3:]
            if request_ids:
                payload["previous_request_ids"] = list(request_ids)
            previous_text_s = str(previous_text or "").strip()
            if previous_text_s:
                payload["previous_text"] = previous_text_s
            next_text_s = str(next_text or "").strip()
            if next_text_s:
                payload["next_text"] = next_text_s
        headers = {
            "xi-api-key": str(api_key),
            "accept": "application/json" if bool(with_timestamps) else "application/octet-stream",
            "content-type": "application/json",
        }
        return str(url), dict(headers), dict(payload), str(output_format)

    @staticmethod
    def _write_pcm16le_wav(path: str, pcm_data: bytes, sample_rate: int = 16000) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        pcm = bytes(pcm_data or b"")
        if len(pcm) % 2 != 0:
            pcm = pcm[: len(pcm) - 1]
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm)
        return path
