#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError
    except Exception as exc:  # pragma: no cover
        print(f"[hf-verify] ERROR: missing huggingface_hub: {exc}", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parents[1]
    load_env(root / "config/worker_secrets.conf")

    token = (
        (os.getenv("AVALIFE_HF_TOKEN") or "").strip()
        or (os.getenv("SMARTBLOG_HF_TOKEN") or "").strip()
        or None
    )
    api = HfApi(token=token)

    worker_repo = (os.getenv("HF_WORKER_REPO_ID") or "abalex2029/smartblog").strip()
    repos = {
        "base model": (
            (os.getenv("HF_BASE_MODEL_REPO_ID") or worker_repo).strip(),
            "Wan2.2-S2V-14B/config.json",
            "Wan2.2-S2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.2-S2V-14B/Wan2.1_VAE.pth",
        ),
        "merged model": (
            (os.getenv("HF_MERGED_MODEL_REPO_ID") or worker_repo).strip(),
            "Wan2.2-S2V-14B-merged-liveavatar-prefp8-test/config.json",
        ),
        "lora": (
            (os.getenv("HF_LORA_REPO_ID") or worker_repo).strip(),
            "LiveAvatar/liveavatar.safetensors",
        ),
        "enhancer models": (
            (os.getenv("HF_ENHANCER_MODELS_REPO_ID") or worker_repo).strip(),
            "enhancers/GFPGANv1.4.pth",
        ),
        "face weights": (
            (os.getenv("HF_FACE_WEIGHTS_REPO_ID") or worker_repo).strip(),
            "enhancers/detection_Resnet50_Final.pth",
            "enhancers/parsing_parsenet.pth",
        ),
    }

    ok = True
    for label, values in repos.items():
        repo_id, *required_files = values
        try:
            files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        except RepositoryNotFoundError:
            print(f"[hf-verify] MISSING REPO: {label}: {repo_id}", file=sys.stderr)
            ok = False
            continue
        except Exception as exc:
            print(f"[hf-verify] ERROR listing {label}: {repo_id}: {exc}", file=sys.stderr)
            ok = False
            continue
        missing = [path for path in required_files if path not in files]
        if missing:
            ok = False
            print(f"[hf-verify] MISSING FILES in {repo_id}:", file=sys.stderr)
            for rel in missing:
                print(f"  - {rel}", file=sys.stderr)
            continue
        print(f"[hf-verify] OK {label}: {repo_id}")
        for rel in required_files:
            print(f"  - {rel}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
