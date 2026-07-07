# Public GitHub Checklist

Use this before changing the repository visibility to public.

## Already Prepared

- Public README links to VlogMe.AI, VlogMe API docs, and the Replicate model.
- Public examples run against Replicate with local sample files.
- `.gitignore` excludes `.env`, `.replicate_runtime/`, model weights, local logs,
  generated outputs, and runtime secret config files.
- Replicate model metadata is synced from `docs/replicate-model-readme.md`.
- The public Replicate schema exposes only `avatar_image`, `audio`, and
  `live_subtitles`.

## Check Before Publishing

1. Confirm `git status --short` is clean.
2. Confirm there are no committed secrets:

   ```bash
   git ls-files | xargs grep -I -n "BEGIN .*PRIVATE KEY\\|ghp_\\|r8_\\|hf_\\|vlm_" || true
   ```

3. Confirm large generated artifacts are not tracked:

   ```bash
   git ls-files | grep -E "^(tmp|logs|output|weights)/" || true
   ```

4. Confirm GitHub Actions secrets exist but are not printed in logs:

   - `REPLICATE_API_TOKEN`
   - `REPLICATE_CLI_AUTH_TOKEN`
   - `VLOGME_API_TOKEN`

## Suggested GitHub Settings

- Description:
  `VlogMe Avatar bridge for Replicate: photo + speech audio to vertical talking-avatar MP4.`
- Website:
  `https://vlogme.ai`
- Topics:
  `replicate`, `cog`, `ai-avatar`, `talking-avatar`, `speech-to-video`,
  `avatar-video`, `ai-video`, `vlogme`
- Social preview:
  use the sample presenter image or a branded VlogMe/Replicate preview.

## After Publishing

1. Open the public repository in a logged-out browser session.
2. Check that README images render.
3. Check that `test_assets/` raw links are accessible.
4. Run the public example from a fresh clone:

   ```bash
   export REPLICATE_API_TOKEN="your_replicate_token"
   python3 examples/run_replicate_prediction.py
   ```

5. Open the public Replicate page and make sure the GitHub link points back to
   the public repository.
