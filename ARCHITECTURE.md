# Architecture

## Goal

Package the VlogMe avatar generator as a standalone Replicate model.

The model should do one thing well:

```text
avatar image + speech audio -> avatar MP4
```

It should not poll VlogMe, claim jobs, write Supabase rows, lease RTX media
workers, call the VlogMe server, or manage B200/B300 commander state.

## Current Bridge

The current runtime already has a useful avatar-only path:

- `avalife.model.main` starts the resident LiveAvatar model runtime.
- `avalife.worker.smartblog_jobs.SmartBlogRenderJobsMixin` prepares avatar
  inputs, one-pass liveaudio chunks, subtitles/watermark/background music, and
  final MP4 files.
- `_smartblog_render_video_job()` returns a `SmartBlogRenderFinalizePlan`.

The Cog predictor uses that render path directly and stops before upload/finalize.

## First Version

Inputs:

- `avatar_image`: required image.
- `audio`: required speech audio.

The first Replicate-facing API intentionally exposes no prompt. The model uses a
small internal default prompt. Later, Gemini can inspect the image and create the
visual prompt automatically inside the model wrapper.

Advanced runtime choices stay environment-driven for now:

- `VLOGME_AVATAR_SIZE_PROFILE=b200|b300`
- `VLOGME_AVATAR_SAMPLE_STEPS`
- `VLOGME_AVATAR_SEED`
- `VLOGME_AVATAR_FACE_RESTORE`
- `VLOGME_AVATAR_BACKGROUND_RESTORE`

Output:

- Local MP4 returned as `cog.Path`.

## Later

- Add optional text-to-speech as a separate mode.
- Add a VlogMe provider adapter that creates Replicate predictions and handles
  webhooks/polling.
- Add CI/CD with private Replicate test model.
- Split the vendored runtime into a real upstream dependency once the interface
  stabilizes.
