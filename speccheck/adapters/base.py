"""Clean adapter interfaces for the three Meta models.

These protocols are the swap point: mock adapters implement them today; the
real Meta adapters in ``meta.py`` will implement the same signatures once the
verified endpoints/params from DOCS_NOTES.md are available. No endpoint,
model ID, or request parameter is assumed anywhere in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..types import ChecklistItemResult, SegmentationResult


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None
    adapter: str = "mock"


class Segmenter(Protocol):
    """SAM 3.1 role: segment an image from text prompts."""

    name: str

    def segment(self, image: np.ndarray, prompts: list[str]) -> SegmentationResult: ...


class Reasoner(Protocol):
    """Muse Spark role: interpret ambiguous items and explain results.

    Deterministic verdicts never come from here; this only turns measured
    values into plain language and flags ambiguous checklist wording.
    """

    name: str

    def explain(self, results: list[ChecklistItemResult], transcript: str | None) -> str: ...


class Transcriber(Protocol):
    """Muse Voice Transcribe role: speech audio -> text command."""

    name: str

    def transcribe(self, audio: bytes, *, filename: str = "audio.webm") -> Transcript: ...
