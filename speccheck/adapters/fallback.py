"""Classical fallback segmenter for real photos (NOT SAM 3.1).

Segments salient foreground blobs and labels them by generic colour.
It only labels a blob when the checklist prompt names that colour, names
the part in the blob's matching prompt, or the spec item carries a
``color:`` hint - otherwise the blob is dropped and the checks honestly
report UNSURE. Confidence is deliberately modest (0.65): this is a
stand-in until the SAM 3.1 adapter lands, and the UI/API label it as such.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from ..types import SegmentedComponent, SegmentationResult
from .mock import _connected_components

GENERIC_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 30),
    "green": (50, 180, 70),
    "blue": (50, 90, 220),
    "yellow": (230, 200, 40),
    "orange": (235, 130, 35),
    "purple": (150, 70, 190),
    "magenta": (210, 60, 170),
    "cyan": (50, 190, 200),
    "gray": (150, 150, 150),
    "grey": (150, 150, 150),
    "white": (235, 235, 235),
    "black": (25, 25, 25),
    "brown": (140, 90, 50),
}


def _nearest_color(mean_rgb: np.ndarray) -> str | None:
    chroma = mean_rgb / max(float(mean_rgb.sum()), 1e-6)
    best: tuple[float, str] | None = None
    for name, rgb in GENERIC_COLORS.items():
        target = np.array(rgb, dtype=float)
        target = target / target.sum()
        dist = float(np.linalg.norm(chroma - target))
        if best is None or dist < best[0]:
            best = (dist, name)
    if best is not None and best[0] <= 0.22:
        return best[1]
    return None


class ClassicalSegmenter:
    """SAM 3.1 substitute: foreground blobs labelled by generic colour."""

    name = "fallback-classical"

    def __init__(self, confidence: float = 0.65) -> None:
        self.confidence = confidence

    def segment(
        self,
        image: np.ndarray,
        prompts: list[str],
        color_hints: dict[str, str] | None = None,
    ) -> SegmentationResult:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("fallback segmenter expects an HxWx3 RGB image")
        arr = np.asarray(
            Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.MedianFilter(3))
        )
        rgb = arr[:, :, :3].astype(float)
        height, width = rgb.shape[:2]
        background = rgb[0, 0]
        foreground = np.linalg.norm(rgb - background, axis=2) > 35.0
        hints = color_hints or {}
        components: list[SegmentedComponent] = []
        for mask in _connected_components(foreground):
            if int(mask.sum()) < 12:
                continue
            color = _nearest_color(rgb[mask].mean(axis=0))
            if color is None:
                continue
            prompt = self._match_prompt(color, prompts, hints)
            if prompt is None:
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

    @staticmethod
    def _match_prompt(
        color: str, prompts: list[str], hints: dict[str, str]
    ) -> str | None:
        for prompt in prompts:
            if hints.get(prompt) == color or color in prompt.lower():
                return prompt
        return None
