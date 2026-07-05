from __future__ import annotations

import os
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_ROOT_DIR = _THIS_FILE.parents[2]
_TORCHRUN_LOG_DIR = Path(os.getenv("TORCHRUN_LOG_DIR", str(_ROOT_DIR / "logs" / "torchrun"))).resolve()
_RUN_STATE_DIR = Path(os.getenv("WORKER_RUN_STATE_DIR", str(_ROOT_DIR / "runtime"))).resolve()
_ACTIVE_RUN_FILE = Path(
    os.getenv("WORKER_ACTIVE_RUN_FILE", str(_RUN_STATE_DIR / "current_torchrun_run.txt"))
).resolve()
_ACTIVE_RUN_LINK = Path(
    os.getenv("WORKER_ACTIVE_RUN_LINK", str(_TORCHRUN_LOG_DIR / "current"))
)


def _detect_current_run_dir() -> Path | None:
    pid = os.getpid()
    for fd in (1, 2):
        try:
            target = Path(os.path.realpath(f"/proc/{pid}/fd/{fd}"))
        except Exception:
            continue
        parts = target.parts
        try:
            log_idx = parts.index("torchrun")
        except ValueError:
            continue
        if log_idx + 1 >= len(parts):
            continue
        run_dir = Path(*parts[: log_idx + 2])
        if run_dir.parent == _TORCHRUN_LOG_DIR and run_dir.is_dir():
            return run_dir
    return None


def record_current_torchrun_run() -> str:
    run_dir = _detect_current_run_dir()
    if run_dir is None:
        return ""

    _RUN_STATE_DIR.mkdir(parents=True, exist_ok=True)
    _TORCHRUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

    tmp_file = _ACTIVE_RUN_FILE.with_suffix(".tmp")
    tmp_file.write_text(f"{run_dir}\n", encoding="utf-8")
    os.replace(tmp_file, _ACTIVE_RUN_FILE)

    try:
        if _ACTIVE_RUN_LINK.is_symlink() or _ACTIVE_RUN_LINK.exists():
            _ACTIVE_RUN_LINK.unlink()
    except FileNotFoundError:
        pass
    _ACTIVE_RUN_LINK.symlink_to(run_dir)
    return str(run_dir)
