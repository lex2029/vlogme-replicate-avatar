# VlogMe Avatar

Create a vertical talking-avatar video from a photo and a speech audio file.

This public Replicate model is the hosted API version of [VlogMe.AI](https://vlogme.ai), built for product demos, explainers, social clips, onboarding videos, sales messages, tutorials, and narrated presenter content. Upload a face or presenter image and a spoken audio track; the output is a vertical 9:16 MP4 video where the person speaks the audio.

You can upload almost any photo. VlogMe uses the center of the image and crops it into a vertical 9:16 frame, so results are best when the face or presenter is near the middle of the image.

If you want the full VlogMe web app with accounts, editing, project history, paid renders, and product updates, visit [vlogme.ai](https://vlogme.ai). If you want to call the avatar generator directly from code, use this Replicate model.

## Inputs

- `avatar_image`: portrait or presenter reference image. The output is always center-cropped to vertical 9:16.
- `audio`: speech audio to animate.
- `live_subtitles`: burn word-level subtitles into the final video. Enabled by default; you can turn it off.

## Watermark policy

Every Replicate generation includes a top watermark that says `Created by VlogMe.AI`.

## Tips for best results

- Use a clear, front-facing or three-quarter portrait.
- Keep the face or presenter near the center of the photo, because the final video is always a vertical 9:16 center crop.
- Avoid tiny faces, heavy occlusion, extreme side profiles, or very low-resolution images.
- Use clean speech audio with minimal music or background noise.
- For subtitles, provide real spoken audio; non-speech tones or music will not produce useful captions.

## Example use cases

- Turn a founder, teacher, creator, or product spokesperson photo into a short talking-avatar video.
- Create API-driven demo videos, onboarding clips, course intros, social posts, and internal training messages.
- Generate vertical presenter clips from your own speech audio or from audio created by another TTS service.

## API notes

The output is a single MP4 file. Predictions can be canceled; active VlogMe render jobs are canceled cooperatively through the bridge.

For a broader VlogMe developer workflow, see the VlogMe API docs at [vlogme.ai/docs/api](https://vlogme.ai/docs/api).

## Limitations

This model is intended for consent-first avatar generation. Do not use it to impersonate people without permission, create misleading identity claims, or generate deceptive content. Results may vary with image quality, audio quality, pose, and lighting.
