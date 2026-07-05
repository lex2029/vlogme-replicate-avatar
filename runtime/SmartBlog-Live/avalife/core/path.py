from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import shutil
import time
from datetime import datetime

from PIL import Image, ImageFilter, ImageOps


def required_worker_fps() -> int:
    raw = str(os.getenv("WORKER_FPS", "") or "").strip()
    if not raw:
        raise RuntimeError("Missing required env: WORKER_FPS")
    try:
        fps = int(raw)
    except Exception as exc:
        raise RuntimeError(f"Invalid WORKER_FPS={raw!r}") from exc
    if fps <= 0:
        raise RuntimeError(f"WORKER_FPS must be > 0, got {fps}")
    return int(fps)


def current_sample_fps(cfg) -> int:
    try:
        fps = int(getattr(cfg, "sample_fps", 0) or 0)
    except Exception:
        fps = 0
    if fps > 0:
        return int(fps)
    return int(required_worker_fps())


def sanitize_job_id(job_id: str) -> str:
    job_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(job_id)).strip("_")
    return job_id[:80] or "job"


def make_job_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return sanitize_job_id(f"{ts}_{random.randint(0, 9999):04d}")


def prepare_live_hls_dir(save_dir: str, job_id: str, *, write_live_player_html) -> str:
    live_dir = os.path.join(save_dir, "live", sanitize_job_id(job_id))
    os.makedirs(live_dir, exist_ok=True)
    for name in os.listdir(live_dir):
        p = os.path.join(live_dir, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except FileNotFoundError:
            pass
    write_live_player_html(live_dir, start_from_beginning=True)
    return live_dir


def prepare_live_raw_dir(save_dir: str, job_id: str) -> str:
    raw_dir = os.path.join(save_dir, "live_raw", sanitize_job_id(job_id))
    os.makedirs(raw_dir, exist_ok=True)
    for name in os.listdir(raw_dir):
        p = os.path.join(raw_dir, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except FileNotFoundError:
            pass
    return raw_dir


def sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def sha256_text(text: str) -> str:
    try:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    except Exception:
        return ""


def persist_idle_image(
    src_path: str,
    *,
    idle_tmp_dir: str | None = None,
    channel_name: str | None = None,
) -> str:
    src_path = (src_path or "").strip()
    if not src_path or not os.path.exists(src_path):
        return src_path
    try:
        idle_tmp_dir = str(idle_tmp_dir or os.getenv("LIVE_IDLE_TMP_DIR", "./tmp/channel_idle") or "./tmp/channel_idle")
        idle_tmp_dir = idle_tmp_dir.strip() or "./tmp/channel_idle"
        channel_name = str(channel_name or os.getenv("LIVE_CHANNEL_NAME", "main") or "main").strip() or "main"
        os.makedirs(idle_tmp_dir, exist_ok=True)
        dst_dir = os.path.join(idle_tmp_dir, str(channel_name))
        os.makedirs(dst_dir, exist_ok=True)
        ext = os.path.splitext(src_path)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            ext = ".jpg"
        sha = sha256_file(src_path) or ""
        if not sha:
            try:
                st = os.stat(src_path)
                sha = f"{int(getattr(st, 'st_mtime_ns', 0) or 0):x}_{int(getattr(st, 'st_size', 0) or 0):x}"
            except Exception:
                sha = str(int(time.time() * 1000.0))
        dst = os.path.join(dst_dir, f"idle_source_{str(sha)[:24]}{ext}")
        if os.path.exists(dst):
            return dst
        tmp = dst + ".tmp"
        shutil.copy2(src_path, tmp)
        os.replace(tmp, dst)
        return dst
    except Exception:
        return src_path


def is_square_size(size: str) -> bool:
    try:
        w, h = map(int, str(size).split("*"))
    except Exception:
        return False
    return w > 0 and h > 0 and w == h


def _env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or default).strip().lower()
    return raw not in {"", "0", "false", "no", "off", "none", "null"}


def _parse_model_size_hw(size: str) -> tuple[int, int] | None:
    raw = str(size or "").strip().lower().replace("x", "*")
    try:
        height_s, width_s = raw.split("*", 1)
        height = int(height_s)
        width = int(width_s)
    except Exception:
        return None
    if int(height) <= 0 or int(width) <= 0:
        return None
    return int(height), int(width)


def _rgb_image_for_ref(image_path: str) -> Image.Image | None:
    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        if img.mode in {"RGBA", "LA"} or "transparency" in getattr(img, "info", {}):
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            bg.alpha_composite(rgba)
            return bg.convert("RGB")
        return img.convert("RGB")
    except Exception:
        return None


def prepare_contain_pad_ref_image(
    image_path: str,
    *,
    target_h: int,
    target_w: int,
    idle_tmp_dir: str,
    channel_name: str,
    prefix: str = "ref_contain",
    job_id: str | None = None,
) -> str:
    image_path = (image_path or "").strip()
    target_h = int(target_h or 0)
    target_w = int(target_w or 0)
    if not image_path or not os.path.exists(image_path) or target_h <= 0 or target_w <= 0:
        return image_path

    img = _rgb_image_for_ref(image_path)
    if img is None:
        return image_path
    src_w, src_h = img.size
    if int(src_w) <= 0 or int(src_h) <= 0:
        return image_path

    sha = sha256_file(image_path) or ""
    if not sha:
        sha = sanitize_job_id(job_id) if job_id else ""
    if not sha:
        try:
            stat = os.stat(image_path)
            sha = f"{int(getattr(stat, 'st_mtime_ns', 0) or 0):x}_{int(getattr(stat, 'st_size', 0) or 0):x}"
        except Exception:
            sha = str(int(time.time() * 1000.0))

    out_dir = os.path.join(str(idle_tmp_dir), str(channel_name), "ref")
    os.makedirs(out_dir, exist_ok=True)
    safe_prefix = sanitize_job_id(str(prefix or "ref_contain"))
    out_path = os.path.join(out_dir, f"{safe_prefix}_{int(target_h)}x{int(target_w)}_{str(sha)[:24]}.png")
    if os.path.exists(out_path):
        return out_path

    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    if int(src_w) == int(target_w) and int(src_h) == int(target_h):
        tmp = out_path + ".tmp"
        img.save(tmp, format="PNG")
        os.replace(tmp, out_path)
        return out_path

    fit_scale = min(float(target_w) / float(src_w), float(target_h) / float(src_h))
    fit_w = max(1, min(int(target_w), int(round(float(src_w) * float(fit_scale)))))
    fit_h = max(1, min(int(target_h), int(round(float(src_h) * float(fit_scale)))))
    foreground = img.resize((int(fit_w), int(fit_h)), resample=resampling)

    cover_scale = max(float(target_w) / float(src_w), float(target_h) / float(src_h))
    cover_w = max(int(target_w), int(round(float(src_w) * float(cover_scale))))
    cover_h = max(int(target_h), int(round(float(src_h) * float(cover_scale))))
    background = img.resize((int(cover_w), int(cover_h)), resample=resampling)
    left = max(0, (int(cover_w) - int(target_w)) // 2)
    top = max(0, (int(cover_h) - int(target_h)) // 2)
    background = background.crop((int(left), int(top), int(left) + int(target_w), int(top) + int(target_h)))
    blur_radius = max(8.0, min(float(min(target_w, target_h)) * 0.04, 28.0))
    background = background.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    paste_x = max(0, (int(target_w) - int(fit_w)) // 2)
    paste_y = max(0, (int(target_h) - int(fit_h)) // 2)
    background.paste(foreground, (int(paste_x), int(paste_y)))

    tmp = out_path + ".tmp"
    background.save(tmp, format="PNG")
    os.replace(tmp, out_path)
    logging.warning(
        "Prepared no-crop avatar ref: src=%s src=%dx%d target=%dx%d fit=%dx%d pad_x=%d pad_y=%d out=%s",
        os.path.basename(str(image_path)),
        int(src_w),
        int(src_h),
        int(target_w),
        int(target_h),
        int(fit_w),
        int(fit_h),
        int(target_w - fit_w),
        int(target_h - fit_h),
        os.path.basename(str(out_path)),
    )
    return out_path


def maybe_square_crop_ref_image(
    image_path: str,
    size: str,
    *,
    idle_tmp_dir: str,
    channel_name: str,
    job_id: str | None = None,
) -> str:
    image_path = (image_path or "").strip()
    if not image_path or not os.path.exists(image_path):
        return image_path
    model_hw = _parse_model_size_hw(str(size))
    if _env_flag("SMARTBLOG_AVATAR_REF_NO_CROP", "1") and model_hw is not None:
        target_h, target_w = model_hw
        return prepare_contain_pad_ref_image(
            image_path,
            target_h=int(target_h),
            target_w=int(target_w),
            idle_tmp_dir=str(idle_tmp_dir),
            channel_name=str(channel_name),
            prefix="ref_nocrop",
            job_id=job_id,
        )
    if not is_square_size(size):
        return image_path

    sha = sha256_file(image_path) or ""
    if not sha:
        sha = sanitize_job_id(job_id) if job_id else ""
    if not sha:
        return image_path

    out_dir = os.path.join(str(idle_tmp_dir), str(channel_name), "ref")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ref_square_{sha}.jpg")
    if os.path.exists(out_path):
        return out_path

    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        w, h = img.size
        side = int(min(w, h))
        if side <= 0:
            return image_path
        left = max(0, (w - side) // 2)
        top = max(0, (h - side) // 2)
        img = img.crop((left, top, left + side, top + side))
        tmp = out_path + ".tmp"
        img.save(tmp, format="JPEG", quality=95)
        os.replace(tmp, out_path)
        return out_path
    except Exception:
        return image_path


def prepare_cover_crop_ref_image(
    image_path: str,
    *,
    target_h: int,
    target_w: int,
    idle_tmp_dir: str | None = None,
    channel_name: str | None = None,
    prefix: str = "ref_cover",
    job_id: str | None = None,
) -> str:
    image_path = (image_path or "").strip()
    target_h = int(target_h or 0)
    target_w = int(target_w or 0)
    if not image_path or not os.path.exists(image_path) or target_h <= 0 or target_w <= 0:
        return image_path

    sha = sha256_file(image_path) or ""
    if not sha:
        sha = sanitize_job_id(job_id) if job_id else ""
    if not sha:
        try:
            stat = os.stat(image_path)
            sha = f"{int(getattr(stat, 'st_mtime_ns', 0) or 0):x}_{int(getattr(stat, 'st_size', 0) or 0):x}"
        except Exception:
            sha = str(int(time.time() * 1000.0))

    idle_tmp_dir = str(idle_tmp_dir or os.getenv("LIVE_IDLE_TMP_DIR", "./tmp/channel_idle") or "./tmp/channel_idle")
    idle_tmp_dir = idle_tmp_dir.strip() or "./tmp/channel_idle"
    channel_name = str(channel_name or os.getenv("LIVE_CHANNEL_NAME", "main") or "main").strip() or "main"
    out_dir = os.path.join(str(idle_tmp_dir), str(channel_name), "ref")
    os.makedirs(out_dir, exist_ok=True)
    safe_prefix = sanitize_job_id(str(prefix or "ref_cover"))
    out_path = os.path.join(out_dir, f"{safe_prefix}_{int(target_h)}x{int(target_w)}_{str(sha)[:24]}.png")
    if os.path.exists(out_path):
        return out_path

    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        if img.mode in {"RGBA", "LA"} or "transparency" in getattr(img, "info", {}):
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            bg.alpha_composite(rgba)
            img = bg.convert("RGB")
        else:
            img = img.convert("RGB")
        src_w, src_h = img.size
        if int(src_w) <= 0 or int(src_h) <= 0:
            return image_path
        scale = max(float(target_w) / float(src_w), float(target_h) / float(src_h))
        new_w = max(int(target_w), int(round(float(src_w) * float(scale))))
        new_h = max(int(target_h), int(round(float(src_h) * float(scale))))
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
        if int(new_w) != int(src_w) or int(new_h) != int(src_h):
            img = img.resize((int(new_w), int(new_h)), resample=resampling)
        left = max(0, (int(new_w) - int(target_w)) // 2)
        top = max(0, (int(new_h) - int(target_h)) // 2)
        img = img.crop((int(left), int(top), int(left) + int(target_w), int(top) + int(target_h)))
        tmp = out_path + ".tmp"
        img.save(tmp, format="PNG")
        os.replace(tmp, out_path)
        return out_path
    except Exception:
        return image_path
