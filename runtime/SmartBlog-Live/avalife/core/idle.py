from __future__ import annotations

import os

def build_idle_prompt_text(prompt: str) -> str:
    _ = prompt
    suffix = str(
        os.getenv(
            "WORKER_IDLE_PROMPT_SUFFIX",
            (
                "The person is silent with lips closed, calm and steady, with very restrained facial expression, "
                "minimal head motion, minimal body motion, only subtle breathing and occasional natural blinking."
            ),
        )
        or ""
    ).strip()
    return suffix
