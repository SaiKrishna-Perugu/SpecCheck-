"""Mock adapters: deterministic, offline stand-ins for the Meta models.

``MockSegmenter`` finds solid-colour blobs (the synthetic sample images are
drawn that way) and matches them to prompts via the shared palette in
:mod:`speccheck.samples`. On arbitrary photos it finds nothing, which the
checks honestly report as UNSURE - it never invents parts.
"""

from __future__ import annotations

import numpy as np

from ..samples import PART_COLORS
from ..types import ChecklistItemResult, SegmentedComponent, SegmentationResult
from .base import Transcript


def _connected_components(foreground: np.ndarray) -> list[np.ndarray]:
    """4-connectivity component masks for a boolean foreground image."""
    height, width = foreground.shape
    visited = np.zeros_like(foreground, dtype=bool)
    components: list[np.ndarray] = []
    ys, xs = np.nonzero(foreground)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        mask = np.zeros_like(foreground, dtype=bool)
        stack = [(y0, x0)]
        visited[y0, x0] = True
        while stack:
            y, x = stack.pop()
            mask[y, x] = True
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and foreground[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        components.append(mask)
    return components


def _nearest_prompt(mean_rgb: np.ndarray, prompts: list[str]) -> str | None:
    best: tuple[float, str] | None = None
    for prompt in prompts:
        for key, rgb in PART_COLORS.items():
            if key in prompt.lower():
                dist = float(np.linalg.norm(mean_rgb - np.array(rgb, dtype=float)))
                if best is None or dist < best[0]:
                    best = (dist, prompt)
    if best is not None and best[0] <= 90.0:
        return best[1]
    return None


class MockSegmenter:
    """SAM 3.1 stand-in: colour-blob segmentation for synthetic images."""

    name = "mock-segmenter"

    def __init__(self, confidence: float = 0.95) -> None:
        self.confidence = confidence

    def segment(self, image: np.ndarray, prompts: list[str]) -> SegmentationResult:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("mock segmenter expects an HxWx3 RGB image")
        rgb = arr[:, :, :3].astype(int)
        height, width = rgb.shape[:2]
        background = tuple(int(v) for v in rgb[0, 0])
        components: list[SegmentedComponent] = []
        colors, counts = np.unique(
            rgb.reshape(-1, 3), axis=0, return_counts=True
        )
        for color, count in zip(colors.tolist(), counts.tolist()):
            if tuple(color) == background or int(count) < 4:
                continue
            color_mask = np.all(rgb == np.array(color), axis=2)
            prompt = _nearest_prompt(np.array(color, dtype=float), prompts)
            if prompt is None:
                continue
            for mask in _connected_components(color_mask):
                if int(mask.sum()) < 4:
                    continue
                components.append(
                    SegmentedComponent(
                        label=prompt,
                        mask=mask,
                        confidence=self.confidence,
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


class MockReasoner:
    """Muse Spark stand-in: template explanations from measured values only."""

    name = "mock-reasoner"

    def explain(self, results: list[ChecklistItemResult], transcript: str | None) -> str:
        parts: list[str] = []
        if transcript:
            parts.append(f'Heard: "{transcript}".')
        for r in results:
            parts.append(f"{r.item_id}: {r.status.value} - {r.explanation}")
        return " ".join(parts)


class MockTranscriber:
    """Muse Voice Transcribe stand-in: no real speech recognition."""

    name = "mock-transcriber"

    def __init__(self, default_text: str = "check the assembly against the spec") -> None:
        self.default_text = default_text

    def transcribe(self, audio: bytes, *, filename: str = "audio.webm") -> Transcript:
        if not audio:
            return Transcript(text="", adapter=self.name)
        # Honest mock: it cannot recognise speech, so it returns the canned
        # command and says so in the text the UI shows.
        return Transcript(
            text=self.default_text,
            adapter=self.name,
        )
