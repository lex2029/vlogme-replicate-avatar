# VlogMe Replicate Avatar

Replicate/Cog package for the VlogMe avatar renderer.

This repo is intentionally separate from the VlogMe app and from the RunPod/Vast
commander architecture. Its first job is narrow: accept one avatar image plus one
speech audio file and return a generated avatar MP4.

## Shape

- `run.py` is the Cog predictor interface.
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
  and face restore disabled. The public `predict()` input also accepts
  `sample_steps`; use `4` for smoke tests and `6+` for quality checks. Tune
  `VLOGME_AVATAR_SIZE`, `VLOGME_AVATAR_SAMPLE_STEPS`, and
  `VLOGME_AVATAR_FACE_RESTORE` after the baseline path is stable.
- A100 acceleration defaults are conservative: BF16/TF32 enabled, merged
  LiveAvatar checkpoint enabled, cuDNN SDPA allowed, external FlashAttention
  disabled, FP8 off, and `torch.compile` off. Compile can be tested with
  `VLOGME_AVATAR_ENABLE_COMPILE=true`; it is restricted to the stable head
  region and skips the dynamic live KV-cache/rope paths.
- Secrets are not needed for the first audio-driven avatar test.

## First Local Test

On a GPU machine with Cog and Docker:

```bash
cd /Users/alekseibabkin/Documents/vlogme-replicate-avatar
mkdir -p weights
cog run -i avatar_image=@/path/to/avatar.png -i audio=@/path/to/speech.wav
```

The first test should use an already prepared speech WAV/MP3. Text-to-speech can
be added later as a separate optional path. The public model interface is kept
simple on purpose: image in, audio in, MP4 out. Prompting is internal for now and
can later be generated from the image with Gemini.

## Replicate Publish

Create a private model on Replicate, then:

```bash
cog push r8.im/<owner>/<model-name>
```

If the local machine does not have Docker/Cog, push this repository to GitHub
and run the `Push to Replicate` workflow. It expects a GitHub Actions secret
named `REPLICATE_CLI_AUTH_TOKEN`.

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
