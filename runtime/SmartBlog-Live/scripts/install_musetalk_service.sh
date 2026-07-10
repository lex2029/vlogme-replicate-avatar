#!/usr/bin/env bash
set -euo pipefail

ROOT="${SMARTBLOG_MUSETALK_ROOT:-/opt/MuseTalk}"
REPO="${SMARTBLOG_MUSETALK_REPO:-https://github.com/TMElyralab/MuseTalk.git}"
REF="${SMARTBLOG_MUSETALK_REF:-main}"
PYTHON_BIN="${SMARTBLOG_MUSETALK_BOOTSTRAP_PYTHON:-$(command -v python3)}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [musetalk-install] $*"; }

mkdir -p "$(dirname "$ROOT")"
if [[ ! -d "$ROOT/.git" ]]; then
  rm -rf "$ROOT"
  log "cloning MuseTalk ref=$REF into $ROOT"
  git clone --depth 1 --branch "$REF" "$REPO" "$ROOT"
else
  log "updating existing MuseTalk checkout at $ROOT"
  git -C "$ROOT" fetch --depth 1 origin "$REF"
  git -C "$ROOT" checkout -q FETCH_HEAD
fi

if [[ ! -x "$ROOT/venv/bin/python" ]]; then
  log "creating venv $ROOT/venv"
  "$PYTHON_BIN" -m venv --system-site-packages "$ROOT/venv"
fi

PIP="$ROOT/venv/bin/python -m pip"
log "installing Python dependencies"
$PIP install -U pip 'setuptools<81' wheel cython
$PIP install \
  'numpy>=1.26,<2.0' \
  'diffusers==0.30.2' \
  'accelerate==0.28.0' \
  'transformers==4.39.2' \
  'huggingface_hub[cli]==0.30.2' \
  'soundfile==0.12.1' \
  'librosa==0.11.0' \
  'einops==0.8.1' \
  'gdown>=5.2' \
  'requests>=2.32' \
  'imageio[ffmpeg]>=2.37' \
  'omegaconf>=2.3' \
  'ffmpeg-python>=0.2' \
  'moviepy>=1.0.3'

if [[ "${SMARTBLOG_MUSETALK_INSTALL_MMPOSE:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  log "installing MMLab runtime dependencies"
  $PIP install 'openmim>=0.3.9'
  "$ROOT/venv/bin/mim" install mmengine
  "$ROOT/venv/bin/mim" install "mmcv>=2.0.1"
  "$ROOT/venv/bin/mim" install "mmdet>=3.1.0"
  "$ROOT/venv/bin/mim" install "mmpose>=1.1.0"
fi

log "patching MuseTalk preprocessing to allow no-MMPose runtime"
"$ROOT/venv/bin/python" "$(dirname "${BASH_SOURCE[0]}")/patch_musetalk_no_mmpose.py" "$ROOT"

log "downloading MuseTalk weights"
(
  cd "$ROOT"
  export HUGGING_FACE_HUB_TOKEN="${SMARTBLOG_HF_TOKEN:-${AVALIFE_HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}}}"
  export HF_TOKEN="${SMARTBLOG_HF_TOKEN:-${AVALIFE_HF_TOKEN:-${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}}"
  "$ROOT/venv/bin/huggingface-cli" download TMElyralab/MuseTalk \
    --local-dir models \
    --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin" "musetalkV15/musetalk.json" "musetalkV15/unet.pth"
  "$ROOT/venv/bin/huggingface-cli" download stabilityai/sd-vae-ft-mse \
    --local-dir models/sd-vae \
    --include "config.json" "diffusion_pytorch_model.bin"
  "$ROOT/venv/bin/huggingface-cli" download openai/whisper-tiny \
    --local-dir models/whisper \
    --include "config.json" "pytorch_model.bin" "preprocessor_config.json"
  if [[ "${SMARTBLOG_MUSETALK_INSTALL_MMPOSE:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    "$ROOT/venv/bin/huggingface-cli" download yzd-v/DWPose \
      --local-dir models/dwpose \
      --include "dw-ll_ucoco_384.pth"
  fi
  "$ROOT/venv/bin/huggingface-cli" download ByteDance/LatentSync \
    --local-dir models/syncnet \
    --include "latentsync_syncnet.pt"
  mkdir -p models/face-parse-bisent
  if [[ ! -s models/face-parse-bisent/79999_iter.pth ]]; then
    "$ROOT/venv/bin/gdown" "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812" -O models/face-parse-bisent/79999_iter.pth
  fi
  if [[ ! -s models/face-parse-bisent/resnet18-5c106cde.pth ]]; then
    curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth -o models/face-parse-bisent/resnet18-5c106cde.pth
  fi
)

log "MuseTalk install complete"
