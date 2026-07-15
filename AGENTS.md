# VlogMe Replicate Avatar Worker Instructions

## Project

This repository publishes the public Replicate bridge model for VlogMe:

- GitHub: `https://github.com/lex2029/vlogme-replicate-avatar`
- Public Replicate model: `lex2029/vlogme-avatar-bridge`
- Bridge Cog config: `cog.bridge.yaml`
- Bridge entrypoint: `run_bridge.py`
- Heavy/private GPU entrypoint: `run.py`

## Secrets

Never print, paste, commit, or log secret values.

Useful local secret/auth locations:

- GitHub token for workflow API calls:
  `/Users/alekseibabkin/.codex/secrets/github_token_lex2029`
- GitHub git credential store may also contain a GitHub token.

The publish workflow already uses GitHub Actions secrets:

- `REPLICATE_CLI_AUTH_TOKEN`
- `REPLICATE_API_TOKEN`
- `VLOGME_API_TOKEN`

Do not copy those secret values into tracked files.

## Publish To Replicate

Preferred publish path is the GitHub Actions workflow:

- Workflow file: `.github/workflows/push-replicate.yml`
- Ref: `main`
- Inputs:
  - `model_name`: `lex2029/vlogme-avatar-bridge`
  - `cog_config`: `cog.bridge.yaml`

If `gh` is unavailable, dispatch the workflow through the GitHub REST API using
the local Codex GitHub token file above. Only print HTTP status, workflow run id,
and public workflow URLs.

After publishing, wait for the workflow run to complete successfully before
updating the Replicate deployment. Publishing creates a new model version but
does not automatically move the deployment to that version.

Then run `.github/workflows/update-replicate-deployment.yml` with:

- `model_name`: `lex2029/vlogme-avatar-bridge`
- `deployment`: `lex2029/vlogme-avatar-bridge-cpu`
- `version`: empty, so the workflow selects the latest model version

Wait for the deployment update to complete successfully before testing. If this
step is skipped, the test can still run the previous worker version.

## Test After Publish

Run an end-to-end Replicate bridge prediction against:

- Model: `lex2029/vlogme-avatar-bridge`
- Deployment: `lex2029/vlogme-avatar-bridge-cpu`

The workflow `.github/workflows/test-replicate-bridge-prediction.yml` is the
preferred hosted test path because it uses GitHub Actions secrets. It should run
until the Replicate prediction completes and returns an MP4.

## Creating VlogMe Bridge Jobs

Read `docs/VLOGME_BRIDGE_JOB_CREATION.md` before changing or recreating the
Replicate-to-VlogMe request. The critical routing field is the top-level JSON
boolean `"replicate_free": true`. Do not omit it, send it as a string, nest it,
or replace it with a caller-supplied `source` field. Without the exact boolean,
VlogMe creates a normal `api_v1` render that can wait on the standard private
render fleet instead of being identified as the public Replicate channel.

Every `POST /api/public/v1/videos` must also include one stable
`Idempotency-Key` for that logical generation. Reuse the same key only when
retrying the same generation; use a new key for a genuinely new generation.

## Cooldown Behavior

The bridge intentionally allows one generation per worker container, then blocks
the next generation for one hour. The local cooldown state is stored under `/tmp`
inside the running Replicate container. If Replicate runs multiple containers,
each container has its own cooldown state.
