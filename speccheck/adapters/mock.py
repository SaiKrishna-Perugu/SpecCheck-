"""Mock adapters: deterministic, offline stand-ins for the Meta models.

``MockSegmenter`` finds solid-colour blobs (the synthetic sample images are
drawn that way) and matches them to prompts via the shared palette in
:mod:`speccheck.samples`. On arbitrary photos it finds nothing, which the
checks honestly report as UNSURE - it never invents parts.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

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


def _chromaticity(rgb: np.ndarray) -> np.ndarray:
    """Brightness-invariant colour: rgb / sum(rgb) per pixel."""
    total = rgb.sum(axis=-1, keepdims=True)
    return rgb / np.maximum(total, 1e-6)


# Max chromaticity distance for a pixel to count as a palette colour.
_MAX_CHROMA_DIST = 0.18


def _prompt_palette(prompts: list[str]) -> dict[str, list[tuple[str, np.ndarray]]]:
    """prompt -> [(palette key, chromaticity)] for keys named in the prompt."""
    out: dict[str, list[tuple[str, np.ndarray]]] = {}
    for prompt in prompts:
        entries = [
            (key, _chromaticity(np.array(rgb, dtype=float).reshape(1, 1, 3))[0, 0])
            for key, rgb in PART_COLORS.items()
            if key in prompt.lower()
        ]
        if entries:
            out[prompt] = entries
    return out


class MockSegmenter:
    """SAM 3.1 stand-in: colour-blob segmentation for synthetic images."""

    name = "mock-segmenter"

    def __init__(self, confidence: float = 0.95) -> None:
        self.confidence = confidence

    def segment(
        self,
        image: np.ndarray,
        prompts: list[str],
        color_hints: dict[str, str] | None = None,
    ) -> SegmentationResult:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("mock segmenter expects an HxWx3 RGB image")
        # 3x3 median denoise: kills speckle noise and JPEG ringing before
        # classification, without moving part boundaries materially.
        arr = np.asarray(
            Image.fromarray(np.asarray(arr, dtype=np.uint8)).filter(
                ImageFilter.MedianFilter(3)
            )
        )
        rgb = arr[:, :, :3].astype(float)
        height, width = rgb.shape[:2]
        background = rgb[0, 0]
        # Foreground = clearly different from the corner/background colour.
        foreground = np.linalg.norm(rgb - background, axis=2) > 35.0
        palette = _prompt_palette(prompts)
        components: list[SegmentedComponent] = []
        if palette and foreground.any():
            chroma = _chromaticity(rgb)
            # Assign each foreground pixel to its nearest palette colour in
            # chromaticity space (robust to noise, JPEG ringing, brightness).
            keys: list[str] = []
            owners: list[str] = []
            targets: list[np.ndarray] = []
            intensities: list[float] = []
            for prompt, entries in palette.items():
                for key, target in entries:
                    keys.append(key)
                    owners.append(prompt)
                    targets.append(target)
                    intensities.append(float(np.mean(PART_COLORS[key])))
            target_arr = np.stack(targets)  # (K, 3)
            intensity_arr = np.array(intensities)
            fg_chroma = chroma[foreground]  # (N, 3)
            fg_intensity = rgb.mean(axis=2)[foreground]
            dists = np.linalg.norm(
                fg_chroma[:, None, :] - target_arr[None, :, :], axis=2
            )
            nearest = dists.argmin(axis=1)
            nearest_dist = dists[np.arange(len(fg_chroma)), nearest]
            fg_ys, fg_xs = np.nonzero(foreground)
            masks: dict[str, np.ndarray] = {
                prompt: np.zeros((height, width), dtype=bool) for prompt in palette
            }
            # Chromaticity alone cannot tell grey parts from grey background
            # noise, so also require a plausible brightness for the colour.
            ratio = fg_intensity / np.maximum(intensity_arr[nearest], 1e-6)
            accepted = (nearest_dist <= _MAX_CHROMA_DIST) & (ratio >= 0.4) & (ratio <= 2.5)
            for y, x, k in zip(
                fg_ys[accepted].tolist(),
                fg_xs[accepted].tolist(),
                nearest[accepted].tolist(),
            ):
                masks[owners[k]][y, x] = True
            for prompt, mask in masks.items():
                for comp_mask in _connected_components(mask):
                    if int(comp_mask.sum()) < 8:
                        continue
                    components.append(
                        SegmentedComponent(
                            label=prompt,
                            mask=comp_mask,
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
