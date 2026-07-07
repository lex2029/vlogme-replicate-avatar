# VlogMe Avatar

Create a talking-avatar video from a portrait image and a speech audio file.

The model is designed for product demos, explainers, social clips, onboarding videos, and narrated presenter content. Upload a clear face image and a spoken audio track; the output is an MP4 video where the person speaks the audio.

## Inputs

- `avatar_image`: portrait or presenter reference image.
- `audio`: speech audio to animate.
- `live_subtitles`: burn word-level subtitles into the final video.
- `watermark_enabled`: keep the VlogMe watermark on for free/default generations.
- `watermark_text`: optional watermark text. Defaults to `Created by VlogMe.AI`.
- `aspect_ratio`: `9:16`, `16:9`, or `1:1`.
- `face_restore`: optional face restoration strength. Leave at `-1` to use VlogMe defaults.
- `video_prompt` and `video_negative_prompt`: optional motion/style guidance.

## Watermark policy

Free/default generations include the VlogMe watermark. Paid/no-watermark workflows can be enabled separately by VlogMe.

## Tips for best results

- Use a clear, front-facing or three-quarter portrait.
- Avoid tiny faces, heavy occlusion, extreme side profiles, or very low-resolution images.
- Use clean speech audio with minimal music or background noise.
- For subtitles, provide real spoken audio; non-speech tones or music will not produce useful captions.

## Limitations

This model is intended for consent-first avatar generation. Do not use it to impersonate people without permission, create misleading identity claims, or generate deceptive content. Results may vary with image quality, audio quality, pose, and lighting.

## API notes

The output is a single MP4 file. Predictions can be canceled; active VlogMe render jobs are canceled cooperatively through the bridge.
