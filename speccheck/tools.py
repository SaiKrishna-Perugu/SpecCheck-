"""Typed tool layer for the SpecCheck agent.

Each tool is a plain typed function. The Meta models sit behind the adapter
protocols in :mod:`speccheck.adapters.base`; swapping the mock adapters for
real ones (see ``speccheck/adapters/meta.py``, TODOs pending DOCS_NOTES.md)
does not change any signature here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from . import checks
from .spec import Checklist, ChecklistItem
from .types import ChecklistItemResult, SegmentationResult


class Segmenter(Protocol):
    """Segments an image from text prompts (SAM 3.1 in production)."""

    name: str

    def segment(
        self, image: np.ndarray, prompts: list[str]
    ) -> SegmentationResult: ...


def segment(
    image: np.ndarray, checklist: Checklist, segmenter: Segmenter
) -> SegmentationResult:
    """Tool: segment target components using text prompts from the checklist."""
    if image is None or image.size == 0:
        raise ValueError("segment: image is empty")
    result = segmenter.segment(image, checklist.prompts())
    mapping = checklist.prompt_to_label()
    for comp in result.components:
        comp.label = mapping.get(comp.prompt, comp.label)
    result.tool = "segment"
    return result


def measure_count(seg: SegmentationResult, label: str) -> int:
    """Tool: count segmented instances of ``label``."""
    return len(seg.by_label(label))


def measure_position(
    seg: SegmentationResult, label: str, reference_label: str, relation: str
) -> dict[str, Any]:
    """Tool: measure whether ``label`` sits in ``relation`` to the reference."""
    item = ChecklistItem(
        id="_adhoc", label=label, check="position",
        relation=relation, reference_label=reference_label,
    )
    result = checks.check_position(item, seg)
    return {"status": result.status.value, **result.measured}


def compare_to_spec(
    checklist: Checklist, seg: SegmentationResult
) -> list[ChecklistItemResult]:
    """Tool: run every deterministic checklist check against the masks."""
    return [checks.run_check(item, seg) for item in checklist.items]


@dataclass
class RecaptureRequest:
    """Tool output asking the user to re-capture, with plain-language reasons."""

    reasons: list[str]
    attempt: int  # 1-based attempt that produced this request
    max_attempts: int = 3

    @property
    def retries_left(self) -> int:
        return max(0, self.max_attempts - self.attempt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasons": list(self.reasons),
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "retries_left": self.retries_left,
        }


def request_recapture(
    results: list[ChecklistItemResult], attempt: int, max_attempts: int = 3
) -> RecaptureRequest | None:
    """Tool: ask for a re-capture when low confidence blocks a verdict.

    Returns ``None`` when no item needs one, or when attempts are exhausted
    (the UNSURE verdicts then stand in the final report).
    """
    reasons = sorted(
        {
            r.recapture_reason
            for r in results
            if r.recapture_reason and r.status.value == "UNSURE"
        }
    )
    if not reasons or attempt >= max_attempts:
        return None
    return RecaptureRequest(reasons=reasons, attempt=attempt, max_attempts=max_attempts)


# build_report lives in speccheck.report (it needs image rendering); it is
# re-exported here so the agent only imports the tool layer.
from .report import Report, build_report  # noqa: E402

__all__ = [
    "Segmenter",
    "segment",
    "measure_count",
    "measure_position",
    "compare_to_spec",
    "RecaptureRequest",
    "request_recapture",
    "Report",
    "build_report",
]
