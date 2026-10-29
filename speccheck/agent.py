"""SpecCheck agent loop.

1. (Voice) transcript arrives from the transcriber adapter.
2. Segment with text prompts from the checklist.
3. Deterministic checks on the masks (no LLM).
4. Reasoner adapter explains in plain language.
5. Low confidence -> request_recapture (max 3 attempts), else final report.
"""

from __future__ import annotations

import numpy as np

from .adapters.base import Reasoner
from .spec import Checklist
from .tools import (
    Segmenter,
    build_report,
    compare_to_spec,
    request_recapture,
    segment,
)
from .report import Report

MAX_ATTEMPTS = 3


def run_check(
    image: np.ndarray,
    checklist: Checklist,
    segmenter: Segmenter,
    reasoner: Reasoner,
    *,
    transcript: str | None = None,
    attempt: int = 1,
    max_attempts: int = MAX_ATTEMPTS,
) -> Report:
    """Run one capture through segment -> measure -> compare -> report."""
    seg = segment(image, checklist, segmenter)
    results = compare_to_spec(checklist, seg)
    explanation = reasoner.explain(results, transcript)
    for result in results:
        if not result.explanation:
            result.explanation = explanation
    recapture = request_recapture(results, attempt=attempt, max_attempts=max_attempts)
    return build_report(
        spec_name=checklist.name,
        results=results,
        image=image,
        transcript=transcript,
        adapter=getattr(segmenter, "name", "unknown"),
        attempt=attempt,
        needs_recapture=recapture is not None,
        recapture_reasons=recapture.reasons if recapture else [],
    )
