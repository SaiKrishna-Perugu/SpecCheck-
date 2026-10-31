"""Real Meta Model API adapters, wired from DOCS_NOTES.md (verified docs).

- SAM 3.1: ``POST /v1/responses``, model ``sam-3.1``, one noun-phrase
  concept per request, masks decoded with ``meta-sam-parser``.
- Muse Spark: ``POST /v1/responses``, model ``muse-spark-1.3``,
  plain-language explanation only (verdicts stay deterministic).
- Muse Voice Transcribe: ``POST /v1/asr/transcribe`` multipart, WAV only.

Transport uses urllib so no SDK dependency is needed. Functions
``_post_json`` / ``_post_multipart`` are the seams unit tests fake.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

import numpy as np
from PIL import Image

from ..types import ChecklistItemResult, SegmentedComponent, SegmentationResult
from .base import Transcript

BASE_URL = "https://api.meta.ai/v1"
SPARK_MODEL = "muse-spark-1.3"
SAM_MODEL = "sam-3.1"
VOICE_MODEL = "muse-voice-transcribe-1.0"
_TIMEOUT_S = 180


def _api_key() -> str:
    key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
    if not key:
        from ..config import get_settings

        get_settings()  # loads the project .env into os.environ
        key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError(
            "META_API_KEY is not set. Copy .env.example to .env and add the key; "
            "never commit it."
        )
    return key


def _request(url: str, data: bytes, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:600]
        raise RuntimeError(f"Meta API HTTP {exc.code} for {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Meta API unreachable ({url}): {exc.reason}") from exc


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    raw = _request(
        f"{BASE_URL}{path}",
        json.dumps(payload).encode("utf-8"),
        {
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
    )
    return json.loads(raw.decode("utf-8"))


def _post_multipart(path: str, fields: dict[str, tuple[str, bytes, str]]) -> dict[str, Any]:
    boundary = "speccheck-boundary-7d1a"
    chunks: list[bytes] = []
    for name, (filename, data, content_type) in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode())
        chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    raw = _request(
        f"{BASE_URL}{path}",
        b"".join(chunks),
        {
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
    )
    return json.loads(raw.decode("utf-8"))


def _output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts: list[str] = []
    for item in response.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "input_text"} and content.get("text"):
                parts.append(str(content["text"]))
    return "".join(parts)


def _image_data_url(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


_BOX_RE = re.compile(
    r"x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?"
)
_MASK_RE = re.compile(r";data=(\d+),(\d+),([~!])(.*)\|>", re.DOTALL)


def parse_sam_output(
    text: str, *, image_width: int, image_height: int, prompt: str
) -> list[SegmentedComponent]:
    """Parse SAM wire text into full-image masks (docs grammar in DOCS_NOTES)."""
    try:
        from meta_sam_parser import SegmentationMask, decode_mask_to_raster
    except ImportError as exc:
        raise RuntimeError(
            "meta-sam-parser is required for SAM masks: uv sync --extra meta"
        ) from exc
    components: list[SegmentedComponent] = []
    starts = [m.start() for m in re.finditer(r"<\|box;", text)]
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(text)
        region = text[start:end]
        box = _BOX_RE.search(region)
        mask_match = _MASK_RE.search(region)
        if box is None or mask_match is None:
            continue
        x1, y1, x2, y2 = (int(box.group(i)) for i in (1, 2, 3, 4))
        confidence = float(box.group(7)) if box.group(7) else 0.5
        mh, mw = int(mask_match.group(1)), int(mask_match.group(2))
        marker = mask_match.group(3)
        payload = marker + mask_match.group(4)
        encoding = "one_bit" if marker == "!" else "lossless"
        raster = np.frombuffer(
            decode_mask_to_raster(
                SegmentationMask(encoding=encoding, payload=payload, width=mw, height=mh)
            ),
            dtype=np.uint8,
        ).reshape(mh, mw)
        full = np.zeros((image_height, image_width), dtype=bool)
        h = min(mh, image_height - y1)
        w = min(mw, image_width - x1)
        if h > 0 and w > 0:
            full[y1 : y1 + h, x1 : x1 + w] = raster[:h, :w] > 0
        if full.any():
            components.append(
                SegmentedComponent(
                    label=prompt, mask=full, confidence=confidence, prompt=prompt
                )
            )
    return components


class MetaSamSegmenter:
    """SAM 3.1 segmenter via the Responses API (one concept per request)."""

    name = "meta-sam-3.1"

    def segment(
        self,
        image: np.ndarray,
        prompts: list[str],
        color_hints: dict[str, str] | None = None,
    ) -> SegmentationResult:
        _api_key()
        arr = np.asarray(image)
        height, width = int(arr.shape[0]), int(arr.shape[1])
        data_url = _image_data_url(arr)
        components: list[SegmentedComponent] = []
        for prompt in prompts:
            response = _post_json(
                "/responses",
                {
                    "model": SAM_MODEL,
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": data_url},
                            ],
                        }
                    ],
                    "stream": False,
                    "metadata": {
                        "include_confidence": "true",
                        "mask_encoding": "one_bit",
                    },
                },
            )
            components.extend(
                parse_sam_output(
                    _output_text(response),
                    image_width=width,
                    image_height=height,
                    prompt=prompt,
                )
            )
        return SegmentationResult(
            image_width=width,
            image_height=height,
            components=components,
            tool="segment",
            adapter=self.name,
        )


class MetaSparkReasoner:
    """Muse Spark explainer. Never changes a deterministic verdict."""

    name = "meta-muse-spark-1.3"

    def explain(self, results: list[ChecklistItemResult], transcript: str | None) -> str:
        _api_key()
        facts = [
            {
                "item": r.item_id,
                "status": r.status.value,
                "measured": r.measured,
                "expected": r.expected,
                "finding": r.explanation,
            }
            for r in results
        ]
        prompt = (
            "You explain visual QA results for a hands-on assembly worker. "
            "Use only the measured facts below; never change a PASS/FAIL/UNSURE "
            "verdict and never invent parts. Two or three short plain sentences. "
            f"Voice command: {transcript or '(none)'}. Facts: {json.dumps(facts)}"
        )
        response = _post_json(
            "/responses",
            {
                "model": SPARK_MODEL,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    }
                ],
                "stream": False,
            },
        )
        text = _output_text(response).strip()
        if not text:
            raise RuntimeError("Muse Spark returned empty explanation text")
        return text


class MetaVoiceTranscriber:
    """Muse Voice Transcribe via POST /v1/asr/transcribe (WAV only)."""

    name = "meta-muse-voice-transcribe-1.0"

    def transcribe(self, audio: bytes, *, filename: str = "audio.webm") -> Transcript:
        _api_key()
        if not audio:
            return Transcript(text="", adapter=self.name)
        wav = _ensure_wav(audio, filename)
        response = _post_multipart(
            "/asr/transcribe",
            {
                "request": (
                    "request.json",
                    json.dumps(
                        {
                            "model": VOICE_MODEL,
                            "audioEncoding": "WAV",
                            "mode": "PUSH_TO_TALK",
                        }
                    ).encode("utf-8"),
                    "application/json",
                ),
                "audio": ("audio.wav", wav, "audio/wav"),
            },
        )
        text = str(response.get("transcript") or "")
        return Transcript(text=text, adapter=self.name)


def _ensure_wav(audio: bytes, filename: str) -> bytes:
    if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        return audio
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError(
            "Muse Voice Transcribe accepts WAV audio only (see DOCS_NOTES.md). "
            f"Received '{filename}' and ffmpeg is not installed to convert it. "
            "Upload a .wav file, install ffmpeg, or type the command instead."
        )
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "input.bin"
        dst = Path(tmp) / "output.wav"
        src.write_bytes(audio)
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0 or not dst.exists():
            raise ValueError("ffmpeg could not convert the recording to WAV")
        return dst.read_bytes()
