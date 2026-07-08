# Examples

This folder contains public examples for running
[`lex2029/vlogme-avatar-bridge`](https://replicate.com/lex2029/vlogme-avatar-bridge).

## Python

```bash
export REPLICATE_API_TOKEN="your_replicate_token"
python3 examples/run_replicate_prediction.py
```

The default example uses the sample inputs committed in `test_assets/`:

- `test_assets/friendly_ai_presenter.jpg`
- `test_assets/presenter_8s.wav`

To verify the free-demo trim with a longer sample:

```bash
python3 examples/run_replicate_prediction.py \
  --audio test_assets/presenter_30s.wav \
  --timeout-sec 2400
```

The public free Replicate demo accepts that file, but renders only its first 10
seconds. Longer renders are available through VlogMe.AI and the paid VlogMe API.

To use your own files:

```bash
python3 examples/run_replicate_prediction.py \
  --image /path/to/portrait.jpg \
  --audio /path/to/speech.wav
```

The script prints the Replicate prediction page and final MP4 URL.
