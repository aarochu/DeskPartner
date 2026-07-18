"""Vision LLM client. Falls back notes: use DESKPARTNER_PERCEPTION=cv."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from p3_vlm_orchestrator.parse import extract_json_object, to_plan
from p3_vlm_orchestrator.prompt import SYSTEM, USER_TEMPLATE
from shared.types import PlanResult


def plan_from_image(frame: np.ndarray, provider: str | None = None) -> PlanResult:
    provider = provider or os.environ.get("DESKPARTNER_VLM_PROVIDER", "anthropic")
    h, w = frame.shape[:2]
    user = USER_TEMPLATE.format(width=w, height=h)

    if provider == "anthropic":
        return _claude(frame, user)
    if provider in {"google", "gemini"}:
        return _gemini(frame, user)
    raise ValueError(f"Unknown provider: {provider}")


def _claude(frame: np.ndarray, user: str) -> PlanResult:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing — set config/.env or use cv mode")
    import base64

    import anthropic

    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("JPEG encode failed")
    b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")
    model = os.environ.get("DESKPARTNER_VLM_MODEL", "claude-sonnet-4-20250514")
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                    },
                    {"type": "text", "text": user},
                ],
            }
        ],
    )
    text = "".join(block.text for block in msg.content if hasattr(block, "text"))
    return to_plan(extract_json_object(text), provider="anthropic")


def _gemini(frame: np.ndarray, user: str) -> PlanResult:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY missing — set config/.env or use cv mode")
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=api_key)
    model_name = os.environ.get("DESKPARTNER_VLM_MODEL", "gemini-2.0-flash")
    model = genai.GenerativeModel(model_name, system_instruction=SYSTEM)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    resp = model.generate_content([user, img])
    return to_plan(extract_json_object(resp.text or "{}"), provider="gemini")


def plan_from_path(path: str | Path, provider: str | None = None) -> PlanResult:
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(path)
    return plan_from_image(frame, provider=provider)
