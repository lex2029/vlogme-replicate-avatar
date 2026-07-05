# VlogMe Replicate Avatar

Replicate/Cog package for the VlogMe avatar renderer.

This repo is intentionally separate from the VlogMe app and from the RunPod/Vast
commander architecture. Its first job is narrow: accept one avatar image plus one
speech audio file and return a generated avatar MP4.

## Shape

- `run.py` is the Cog interface.
- `runtime/SmartBlog-Live/` is a vendored snapshot of the current avatar runtime.
- Model weights are not committed. Put them under `weights/` locally, or set
  `VLOGME_AVATAR_ASSET_ROOT=/path/to/assets`.
- On first start, `VLOGME_AVATAR_PRESEED_MODE` defaults to `verify-or-preseed`,
  so a fresh container will verify local weights and then try to download missing
  assets.
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
warm for subsequent `run()` calls. `run()` routes through the existing
avatar-only render path but bypasses VlogMe job polling and Supabase upload.

The output is returned to Replicate as a local file. VlogMe should upload/store
the result from its own backend after receiving the Replicate prediction output.
