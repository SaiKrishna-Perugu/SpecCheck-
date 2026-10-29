"""Shared typed data model for SpecCheck.

Masks are 2-D ``numpy.bool_`` arrays with the same height/width as the source
image. Bounding boxes and centroids are in image pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNSURE = "UNSURE"


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixels: (x_min, y_min) top-left."""

    x_min: int
    y_min: int
    x_max: int  # exclusive
    y_max: int  # exclusive

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    def to_dict(self) -> dict[str, int]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }


def bbox_of_mask(mask: np.ndarray) -> BBox | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def centroid_of_mask(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


@dataclass
class SegmentedComponent:
    """One segmented instance produced by a segmenter for a text prompt."""

    label: str
    mask: np.ndarray  # bool array, image-sized
    confidence: float  # 0..1, segmenter-reported
    prompt: str = ""

    @property
    def area_px(self) -> int:
        return int(self.mask.sum())

    @property
    def bbox(self) -> BBox | None:
        return bbox_of_mask(self.mask)

    @property
    def centroid(self) -> tuple[float, float] | None:
        return centroid_of_mask(self.mask)

    def area_fraction(self, image_area_px: int) -> float:
        if image_area_px <= 0:
            return 0.0
        return self.area_px / float(image_area_px)

    def to_dict(self, *, include_mask_rle: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "label": self.label,
            "prompt": self.prompt,
            "confidence": round(float(self.confidence), 4),
            "area_px": self.area_px,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "centroid": (
                [round(float(self.centroid[0]), 2), round(float(self.centroid[1]), 2)]
                if self.centroid
                else None
            ),
        }
        if include_mask_rle:
            data["mask_rle"] = mask_to_rle(self.mask)
        return data


@dataclass
class SegmentationResult:
    image_width: int
    image_height: int
    components: list[SegmentedComponent] = field(default_factory=list)
    tool: str = "segment"
    adapter: str = "mock"

    @property
    def image_area_px(self) -> int:
        return self.image_width * self.image_height

    def by_label(self, label: str) -> list[SegmentedComponent]:
        return [c for c in self.components if c.label == label]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "tool": self.tool,
            "adapter": self.adapter,
            "components": [c.to_dict() for c in self.components],
        }


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Row-major run-length encoding of a bool mask (JSON-safe evidence)."""
    flat = np.asarray(mask, dtype=bool).ravel()
    runs: list[list[int]] = []
    value = False
    count = 0
    for px in flat:
        if bool(px) == value:
            count += 1
        else:
            runs.append([1 if value else 0, count])
            value = bool(px)
            count = 1
    if count:
        runs.append([1 if value else 0, count])
    return {"shape": [int(mask.shape[0]), int(mask.shape[1])], "runs": runs}


def mask_from_rle(data: dict[str, Any]) -> np.ndarray:
    h, w = int(data["shape"][0]), int(data["shape"][1])
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    for value, count in data["runs"]:
        count = int(count)
        if int(value) == 1:
            flat[pos : pos + count] = True
        pos += count
    return flat.reshape(h, w)


@dataclass
class Evidence:
    """Evidence linked to one checklist verdict. No evidence -> UNSURE."""

    tool: str
    components: list[SegmentedComponent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_evidence(self) -> bool:
        return len(self.components) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "components": [c.to_dict() for c in self.components],
            "notes": list(self.notes),
        }


@dataclass
class ChecklistItemResult:
    item_id: str
    label: str
    check: str
    status: CheckStatus
    measured: dict[str, Any]
    expected: dict[str, Any]
    evidence: Evidence
    explanation: str = ""
    recapture_reason: str | None = None  # set when status is UNSURE and retry may help

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "label": self.label,
            "check": self.check,
            "status": self.status.value,
            "measured": self.measured,
            "expected": self.expected,
            "evidence": self.evidence.to_dict(),
            "explanation": self.explanation,
            "recapture_reason": self.recapture_reason,
        }
