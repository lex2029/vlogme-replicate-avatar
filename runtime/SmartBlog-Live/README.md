<p align="center">
  <img src="assets/github-banner.svg" alt="VlogMe render worker banner" width="100%" />
</p>

<h1 align="center">VlogMe Render Worker</h1>

<p align="center">
  Render worker runtime for <b>VlogMe</b>.
  <br />
  B200 avatar commander plus RTX media/finalizer worker for mixed render timelines.
</p>

<p align="center">
  <a href="docs/B300_RENDER_ONLY_PRUNE_PLAN_RU.md"><b>Render Invariants</b></a>
  ·
  <a href="docs/worker-contract-render.md"><b>Render Contract</b></a>
  ·
  <a href="docs/VLOGME_RUNPOD_RENDER_FLEET_RU.md"><b>VlogMe RunPod Fleet</b></a>
  ·
  <a href="docs/DEPLOY_RUNPOD_RENDER_RU.md"><b>RunPod Render Deploy</b></a>
  ·
  <a href="config/README.md"><b>Config Layout</b></a>
  ·
  <a href="docs/DEPLOY_CONTAINER_PROD_RU.md"><b>Container Deploy</b></a>
</p>

## What This Repo Runs

This branch is the active render-worker line and no longer an upstream
`LiveAvatar` mirror.

It runs:
- VlogMe `render_video` file-render jobs.
- Avatar-only videos on the B200 commander.
- Mixed Avatar+Hunyuan timelines where B200 owns LiveAvatar/timeline planning
  and RTX media workers own Hunyuan, MMAudio and final postprocessing.
- Direct VlogMe short Hunyuan render jobs on the RTX media worker when the
  VlogMe render poller is enabled.

It does not claim live jobs. Live remains on the separate 3xB200 live-worker
line.

## Core Principles

- B200 is an avatar commander, not a local Hunyuan/MMAudio host.
- RTX media workers own Hunyuan, MMAudio and final postprocessing.
- One generated active profile plus one generated local edge profile.
- One checked-in profile override file for hardware role differences.
- No split producer/edge profile overlays in this branch.
- No hidden host-local runtime overrides.

## Runtime Map

| Surface | Runtime Path | Behavior |
| --- | --- | --- |
| Avatar-only `render_video` | LiveAvatar + local edge | one-pass avatar layers where possible |
| Hunyuan `render_video` | remote RTX media service | Hunyuan clips/inserts |
| Mixed `render_video` | B200 planner + remote RTX media | avatar layer, Hunyuan layer, avatar continuation layers |
| Postprocess | remote RTX finalizer | RIFE/VFX/upscale/final encode |
| VlogMe direct Hunyuan render | RTX media poller | Hunyuan 16fps + MMAudio + finalizer to 30fps |

## Repo Layout

```text
/root/SmartBlog-Live
├── avalife/
│   ├── core/      # shared runtime and generation core
│   ├── model/     # model runtime server and media process
│   └── worker/    # frontend worker, LiveKit and API control plane
├── config/        # locked config, runtime config, render profiles
├── deploy/        # systemd and container deployment files
├── docs/          # current operational docs
├── liveavatar/    # model-side runtime and vendor code
├── scripts/       # bootstrap, profile switching, service control
└── tests/         # runtime and API parity tests
```

Project layout stays self-contained:
- code checkout: `/root/SmartBlog-Live`
- downloaded models and enhancer assets: inside `/root/SmartBlog-Live`
- runtime state, watchdog logs and sockets: inside `/root/SmartBlog-Live/runtime`
- archived old repos/backups: `/root/old/...`
- stable `systemd` anchor: `/root/smartblog-live-current -> /root/SmartBlog-Live`

## Quick Start

For a clean render container:

```bash
cd /root/SmartBlog-Live
./scripts/profile.sh b200-avatar-commander
# or, on a B300 primary render pod:
./scripts/profile.sh b300-avatar-commander
```

RunPod/Vast container deploy uses `Dockerfile.b200-avatar-commander` and
`scripts/runpod_b200_render_start.sh`.

VlogMe Worker API uses:
- `WORKER_API_KEY` for Worker API REST calls
- `SUPABASE_SERVICE_ROLE_KEY` for Supabase Realtime WebSocket
- `SUPABASE_URL` for the Supabase project root

Runtime credentials are loaded from `config/worker_secrets.conf`. In this
private-repo deployment mode the file is intentionally tracked so worker hosts
can update credentials with the same `git pull` flow as code updates.

The runtime install includes pinned restore dependencies required by live/video generation:
- `basicsr`
- `facexlib`
- `gfpgan`
- `realesrgan`

## Runtime Control

Systemd is the primary control path:

```bash
systemctl status smartblog-live-modeld.service
systemctl status smartblog-live-frontend.service
systemctl status smartblog-live-watchdog.service
cd /root/SmartBlog-Live
./scripts/control.sh status
./scripts/watchdog.sh status
```

If `systemd` is unavailable or you do not want to use it, the same watchdog can run locally in the background:

```bash
cd /root/SmartBlog-Live
WORKER_WATCHDOG_MODE=local ./scripts/watchdog.sh start
WORKER_WATCHDOG_MODE=local ./scripts/watchdog.sh status
WORKER_WATCHDOG_MODE=local ./scripts/watchdog.sh stop
```

## Render Profiles

This branch uses generated active profiles for render workers:

- `config/worker_profile.local.conf`
- `config/worker_profile.edge.local.conf`

Checked-in sources:

- `config/worker_profile.render_allinone.conf`
- `config/worker_profile.render_edge.conf`
- `config/worker_profile.render_overrides.conf`

Switch profile:

```bash
cd /root/SmartBlog-Live
./scripts/profile.sh show
./scripts/profile.sh b200-avatar-commander --restart
./scripts/profile.sh b300-avatar-commander --restart
./scripts/profile.sh rtxpro6000-media --restart
```

Meaning:
- `b200-avatar-commander`: VlogMe render-video commander; owns avatar generation and timeline planning.
- `b300-avatar-commander`: same split commander path, but with the larger B300 avatar canvas.
- `rtxpro6000-media`: VlogMe media worker; owns Hunyuan, MMAudio and finalization/upscale.
- Live remains on the separate live worker line.

## Weight Verification

```bash
cd /root/SmartBlog-Live
./.venv/bin/python scripts/verify_worker_weights_hf.py
WORKER_ASSET_ROOT=/root/SmartBlog-Live bash scripts/download_worker_weights.sh
```

## Current Documentation

- [docs/VLOGME_RUNPOD_RENDER_FLEET_RU.md](docs/VLOGME_RUNPOD_RENDER_FLEET_RU.md)
- [docs/B300_RENDER_ONLY_PRUNE_PLAN_RU.md](docs/B300_RENDER_ONLY_PRUNE_PLAN_RU.md)
- [docs/DEPLOY_RUNPOD_RENDER_RU.md](docs/DEPLOY_RUNPOD_RENDER_RU.md)
- [docs/DEPLOY_CONTAINER_PROD_RU.md](docs/DEPLOY_CONTAINER_PROD_RU.md)

## Notes

- Live sessions intentionally keep their own non-interrupting live/comment behavior.
- Weights and enhancer assets are expected inside the project folder under `WORKER_ASSET_ROOT` and are ignored by git.
- This repo is meant to stay clean and product-focused. Old tools, demos and legacy side paths belong in `/root/old`, not here.
