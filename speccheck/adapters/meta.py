"""Real Meta Model API adapters - NOT wired yet, by design.

Step 0 (DOCS_NOTES.md) is paused: the user will provide the verified APIs.
Until then nothing here invents an endpoint, model ID, or parameter. Each
method raises ``NotImplementedError`` with the exact TODO to fill in once
DOCS_NOTES.md confirms it. The mock adapters are the runnable path today.
"""

from __future__ import annotations

import os

import numpy as np

from ..types import ChecklistItemResult, SegmentationResult
from .base import Transcript

_TODO = (
    "TODO(DOCS_NOTES): {what} is not wired yet. Fill this in from the verified "
    "Meta Model API docs (exact model ID, endpoint, image/audio encoding, and "
    "response shape) - do not guess. See DOCS_NOTES.md."
)


def _require_key() -> str:
    key = os.environ.get("META_API_KEY")
    if not key:
        raise RuntimeError(
            "META_API_KEY is not set. Copy .env.example to .env and add the key; "
            "never commit it."
        )
    return key


class MetaSamSegmenter:
    """SAM 3.1 segmenter (real API pending DOCS_NOTES.md)."""

    name = "meta-sam"  # TODO(DOCS_NOTES): confirm exact model ID.

    def segment(self, image: np.ndarray, prompts: list[str]) -> SegmentationResult:
        _require_key()
        raise NotImplementedError(
            _TODO.format(what="SAM 3.1 text-prompt segmentation")
        )


class MetaSparkReasoner:
    """Muse Spark reasoner (real API pending DOCS_NOTES.md)."""

    name = "meta-muse-spark"  # TODO(DOCS_NOTES): confirm exact model ID.

    def explain(self, results: list[ChecklistItemResult], transcript: str | None) -> str:
        _require_key()
        raise NotImplementedError(
            _TODO.format(what="Muse Spark multimodal explanation / tool choice")
        )


class MetaVoiceTranscriber:
    """Muse Voice Transcribe (real API pending DOCS_NOTES.md)."""

    name = "meta-muse-voice"  # TODO(DOCS_NOTES): confirm exact model ID.

    def transcribe(self, audio: bytes, *, filename: str = "audio.webm") -> Transcript:
        _require_key()
        raise NotImplementedError(
            _TODO.format(what="Muse Voice Transcribe speech-to-text")
        )
