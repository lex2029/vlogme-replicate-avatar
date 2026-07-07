# VlogMe Replicate Avatar

Replicate/Cog package for the VlogMe avatar renderer and the lightweight
Replicate-to-VlogMe bridge.

This repo is intentionally separate from the VlogMe app and from the RunPod/Vast
commander architecture. Its first job is narrow: accept one avatar image plus one
speech audio file and return a generated avatar MP4.

## Shape

- `run.py` is the heavy GPU Cog predictor interface.
- `run_bridge.py` is the lightweight bridge predictor. It accepts the same
  avatar image plus speech audio shape, creates a VlogMe public API render job,
  waits for our worker to finish it, downloads the completed MP4, and returns
  that file to Replicate.
- `cog.yaml` builds the heavy A100 avatar image.
- `cog.bridge.yaml` builds the cheap CPU bridge image.
- `runtime/SmartBlog-Live/` is a vendored snapshot of the current avatar runtime.
- Model weights are not committed. Put them under `weights/` locally, or set
  `VLOGME_AVATAR_ASSET_ROOT=/path/to/assets`.
- On first start, `VLOGME_AVATAR_PRESEED_MODE` defaults to `verify-or-preseed`,
  so a fresh container will verify local weights and then try to download missing
  assets.
- The default runtime profile targets Replicate `gpu-a100-large-2x`:
  `VLOGME_AVATAR_GPU_LAYOUT=passthrough`. Passthrough runs DiT/denoise on GPU 0
  and keeps GPU 1 dedicated to VAE/decode/stream-file/post-VAE work. It uses
  32-frame inference windows, 4 latent frames per Wan block, and a 32-frame KV
  cache so the denoise side keeps one full output window of history. Use
  `VLOGME_AVATAR_GPU_LAYOUT=dit2` only for explicit A/B tests where both A100
  cards should shard DiT/denoise. The default first-pass Replicate canvas is
  portrait `704*384` (Wan/model order is height*width), with 6 inference steps
  and PostVAE/GFPGAN face restore disabled. When face restore is explicitly
  enabled, file rendering restores faces on the PostVAE x2 layer and uses the
  official GFPGAN/enchenh2d crop-paste path instead of the experimental batch
  overlay path. The B300-style native-first path can still be A/B tested with
  `face_restore_stage=native_first`, and the old batch paste path with
  `LIVE_RAW_POST_VAE_FACE_PASTE_MODE=batch`.
  The public `predict()` input also accepts
  `sample_steps`; use `4` for smoke tests and `6+` for quality checks. Tune
  `VLOGME_AVATAR_SIZE`, `VLOGME_AVATAR_SAMPLE_STEPS`, and
  `VLOGME_AVATAR_FACE_RESTORE` after the baseline path is stable. For GFPGAN
  diagnostics only, pass `debug_face_crops=1`; the prediction returns a zip
  containing `avatar.mp4` plus aligned/restored/composited face crop JPEGs.
- A100 acceleration defaults are conservative: BF16/TF32 enabled, merged
  LiveAvatar checkpoint enabled, cuDNN SDPA allowed, external FlashAttention
  disabled, FP8 off, and `torch.compile` off. Compile can be tested with
  `VLOGME_AVATAR_ENABLE_COMPILE=true`; it is restricted to the stable head
  region and skips the dynamic live KV-cache/rope paths.
- The heavy GPU test only needs `hf_token` if runtime weights must be pulled
  from Hugging Face. The bridge needs `VLOGME_API_TOKEN` configured in the
  Replicate deployment environment. Until Replicate runtime secrets are
  configured for the deployment, the predictor keeps an optional
  `vlogme_api_token` Secret input as a private smoke-test fallback. Do not bake
  that token into the image.

## First Local Test

On a GPU machine with Cog and Docker:

```bash
cd /Users/alekseibabkin/Documents/vlogme-replicate-avatar
mkdir -p weights
cog run -i avatar_image=@/path/to/avatar.png -i audio=@/path/to/speech.wav
```

The first test should use an already prepared speech WAV/MP3. Text-to-speech can
be added later as a separate optional path. The public bridge interface is kept
simple on purpose: image in, audio in, optional subtitles toggle, MP4 out.
Prompting is internal for now and can later be generated from the image with
Gemini. The bridge always requests a vertical 9:16 render. VlogMe accepts almost
any reference photo, then uses the center of the image for the vertical crop, so
faces/presenters should be near the middle. Replicate bridge generations always
include the top watermark `Created by VlogMe.AI`.

## Replicate Publish

Create a private model on Replicate, then:

```bash
cog push r8.im/<owner>/<model-name>
```

If the local machine does not have Docker/Cog, push this repository to GitHub
and run the `Push to Replicate` workflow. It expects a GitHub Actions secret
named `REPLICATE_CLI_AUTH_TOKEN`.

For the bridge image, use the same workflow with:

- `model_name`: for example `lex2029/vlogme-avatar-bridge`
- `cog_config`: `cog.bridge.yaml`

The bridge smoke workflow is `Test Replicate Bridge Prediction`. It expects:

- `REPLICATE_API_TOKEN` for the Replicate API.
- A Replicate deployment, for example `lex2029/vlogme-avatar-bridge-cpu`, with
  `VLOGME_API_TOKEN` configured as an environment secret. The bridge submits
  through `POST /api/public/v1/videos` and polls
  `GET /api/public/v1/videos/:id`.
- The GitHub Actions secret `VLOGME_API_TOKEN` is only needed when the smoke
  workflow submits private bridge smoke predictions or verifies cooperative
  cancellation against the VlogMe API. Once the Replicate deployment provides
  `VLOGME_API_TOKEN` at runtime, remove the fallback Secret input from the public
  schema.
- For cancellation handoff, pass the VlogMe Replicate webhook URL when creating
  predictions:
  `https://vlogme.ai/api/public/v1/replicate/webhook`, with events
  `logs,completed`. The smoke workflow exposes this as `webhook_url` and
  `webhook_events`.
- The bridge exposes `live_subtitles` as the only user-facing render toggle.
  Subtitles are on by default and can be disabled. Aspect ratio is always
  vertical `9:16`, and the top watermark is always `Created by VlogMe.AI`.

The `Cancel Active Replicate Predictions` workflow scans recent Replicate
predictions and cancels active `starting`/`processing` jobs for a model or
deployment. Use it before switching from the heavy A100 deployment to the bridge
deployment if a long render is still running.

Use a deployment for production traffic so min/max instances, hardware, and
rolling updates are controlled outside the VlogMe app.

## Runtime Notes

The wrapper starts the LiveAvatar model runtime once in `setup()` and keeps it
warm for subsequent `predict()` calls. `predict()` now calls the model runtime
directly with a single `InferRequest` and direct stream-file MP4 output. It does
not create a SmartBlog render job, claim, mock state, VlogMe job poll, Supabase
upload, Hunyuan, MMAudio, or remote edge handoff.

The output is returned to Replicate as a local file. VlogMe should upload/store
the result from its own backend after receiving the Replicate prediction output.
