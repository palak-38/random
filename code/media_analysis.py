from __future__ import annotations

import json
import re
from pathlib import Path

from config import CACHE_DIR, DATASET_DIR, GEMINI_MODEL, MEDIA_ANALYSIS_CACHE, gemini_api_key
from schemas import MediaAnalysis, MediaContent

URL_RE = re.compile(r"\b(?:https?://)?[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)+(?:/[^\s,]*)?", re.I)
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?\d{10}\b")
PAYMENT_RE = re.compile(r"\b(pay|payment|upi|rs\.?|inr|₹|fee|charge|refund|wallet|bank)\b", re.I)

IMAGE_PROMPT = (
    "You are analysing an image sent over WhatsApp. Reply with strict JSON only, no markdown fences:\n"
    '{"description": "<one sentence on what this image is, e.g. sale poster, school notice, screenshot of a payment page>", '
    '"text": "<all readable text in the image, verbatim; empty string if none>"}'
)

VOICE_PROMPT = (
    "Transcribe this voice note verbatim. Reply with strict JSON only, no markdown fences:\n"
    '{"description": "<one sentence on tone and intent, e.g. calm family update, urgent request>", '
    '"text": "<verbatim transcript>"}'
)


def _load_cache() -> dict[str, dict]:
    if MEDIA_ANALYSIS_CACHE.exists():
        return json.loads(MEDIA_ANALYSIS_CACHE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict[str, dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_ANALYSIS_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _derive(media_id: str, media_type: str, description: str, text: str) -> MediaAnalysis:
    blob = f"{description}\n{text}"
    return MediaAnalysis(
        media_id=media_id,
        media_type=media_type,
        transcript=text.strip(),
        description=description.strip(),
        detected_urls=sorted({m.group(0) for m in URL_RE.finditer(blob) if "." in m.group(0)}),
        detected_phone_numbers=sorted({m.group(0) for m in PHONE_RE.finditer(blob)}),
        mentions_payment=bool(PAYMENT_RE.search(blob)),
    )


def _parse_json(raw: str) -> tuple[str, str]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        data = json.loads(cleaned)
        return str(data.get("description", "")), str(data.get("text", ""))
    except json.JSONDecodeError:
        return "", cleaned


def analyse_all(media_items: list[MediaContent], force: bool = False) -> dict[str, MediaAnalysis]:
    """One-time Gemini pass over every media file. Results are cached to disk so
    reruns never re-call the API."""
    cache = {} if force else _load_cache()
    # A cached error is not a result: retry it, but keep genuine "file missing" verdicts.
    todo = [
        m
        for m in media_items
        if m.media_id not in cache
        or (cache[m.media_id].get("analysis_error") and cache[m.media_id]["analysis_error"] != "file missing")
    ]

    if todo:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=gemini_api_key())
        for item in todo:
            path = Path(DATASET_DIR / item.file_path)
            if not path.exists():
                cache[item.media_id] = MediaAnalysis(
                    media_id=item.media_id,
                    media_type=item.media_type,
                    analysis_error="file missing",
                ).model_dump()
                continue

            prompt = IMAGE_PROMPT if item.media_type == "image" else VOICE_PROMPT
            mime = "image/jpeg" if item.media_type == "image" else "audio/mpeg"
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        types.Part.from_bytes(data=path.read_bytes(), mime_type=mime),
                        prompt,
                    ],
                )
                description, text = _parse_json(response.text or "")
                analysis = _derive(item.media_id, item.media_type, description, text)
            except Exception as exc:  # noqa: BLE001 - cached so the run can continue
                analysis = MediaAnalysis(
                    media_id=item.media_id,
                    media_type=item.media_type,
                    analysis_error=str(exc)[:200],
                )
            cache[item.media_id] = analysis.model_dump()
            print(f"  analysed {item.media_id}")
            _save_cache(cache)

    return {k: MediaAnalysis.model_validate(v) for k, v in cache.items()}


def load_cached() -> dict[str, MediaAnalysis]:
    return {k: MediaAnalysis.model_validate(v) for k, v in _load_cache().items()}
