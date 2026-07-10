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
testing the Replicate model.

## Test After Publish

Run an end-to-end Replicate bridge prediction against:

- Model: `lex2029/vlogme-avatar-bridge`
- Deployment: `lex2029/vlogme-avatar-bridge-cpu`

The workflow `.github/workflows/test-replicate-bridge-prediction.yml` is the
preferred hosted test path because it uses GitHub Actions secrets. It should run
until the Replicate prediction completes and returns an MP4.

## Cooldown Behavior

The bridge intentionally allows one generation per worker container, then blocks
the next generation for one hour. The local cooldown state is stored under `/tmp`
inside the running Replicate container. If Replicate runs multiple containers,
each container has its own cooldown state.
