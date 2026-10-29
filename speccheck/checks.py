"""Deterministic checks on segmentation masks.

No LLM is used here: counts, relative positions, area ratios and alignment
are computed directly from mask geometry. A verdict is only PASS/FAIL when
there is confident evidence; anything else is UNSURE (never PASS).
"""

from __future__ import annotations

from typing import Any

from .spec import ChecklistItem
from .types import (
    ChecklistItemResult,
    CheckStatus,
    Evidence,
    SegmentedComponent,
    SegmentationResult,
)


def _usable(
    item: ChecklistItem, seg: SegmentationResult, label: str
) -> tuple[list[SegmentedComponent], str | None]:
    """Split components for ``label`` into usable evidence vs a recapture reason."""
    comps = seg.by_label(label)
    if not comps:
        return [], f"no '{label}' was segmented - it may be missing, occluded, or out of frame"
    good = [
        c
        for c in comps
        if c.confidence >= item.min_confidence
        and c.area_fraction(seg.image_area_px) >= item.min_area_fraction
    ]
    if good:
        return good, None
    low_conf = [c for c in comps if c.confidence < item.min_confidence]
    tiny = [c for c in comps if c.area_fraction(seg.image_area_px) < item.min_area_fraction]
    reasons: list[str] = []
    if low_conf:
        reasons.append(
            f"segmentation confidence {max(c.confidence for c in low_conf):.2f} "
            f"is below {item.min_confidence:.2f} (possible occlusion or bad angle)"
        )
    if tiny:
        reasons.append("segmented mask is very small (part may be too far away or partly hidden)")
    return [], "; ".join(reasons) or f"'{label}' evidence is not usable"


def _result(
    item: ChecklistItem,
    status: CheckStatus,
    measured: dict[str, Any],
    expected: dict[str, Any],
    evidence: Evidence,
    explanation: str,
    recapture_reason: str | None = None,
) -> ChecklistItemResult:
    # Invariant: without evidence a claim can never be PASS.
    if status == CheckStatus.PASS and not evidence.has_evidence:
        status = CheckStatus.UNSURE
        recapture_reason = recapture_reason or "no evidence was captured for this item"
    return ChecklistItemResult(
        item_id=item.id,
        label=item.label,
        check=item.check,
        status=status,
        measured=measured,
        expected=expected,
        evidence=evidence,
        explanation=explanation,
        recapture_reason=recapture_reason if status == CheckStatus.UNSURE else None,
    )


def check_count(item: ChecklistItem, seg: SegmentationResult) -> ChecklistItemResult:
    expected = {"expected_count": item.expected_count}
    comps, reason = _usable(item, seg, item.label)
    all_comps = seg.by_label(item.label)
    measured = {"count": len(comps), "candidates_seen": len(all_comps)}
    ev = Evidence(tool="measure_count", components=comps)
    if reason is not None and not comps:
        # Zero confident instances: FAIL only when other labels segmented fine
        # (image was readable); with no signal at all stay UNSURE.
        if all_comps:
            return _result(
                item, CheckStatus.UNSURE, measured, expected, ev,
                f"Found {len(all_comps)} candidate(s) for '{item.label}' but none were "
                "confident enough to count.", reason,
            )
        if seg.components:
            return _result(
                item, CheckStatus.FAIL, measured, expected, ev,
                f"Expected {item.expected_count} x '{item.label}' but found none, while "
                "other parts were visible.", None,
            )
        return _result(
            item, CheckStatus.UNSURE, measured, expected, ev,
            "Nothing could be segmented in this image.", reason,
        )
    if len(comps) == item.expected_count:
        return _result(
            item, CheckStatus.PASS, measured, expected, ev,
            f"Found {len(comps)} x '{item.label}', matching the spec.", None,
        )
    return _result(
        item, CheckStatus.FAIL, measured, expected, ev,
        f"Expected {item.expected_count} x '{item.label}' but found {len(comps)} "
        "(missing or extra parts).", None,
    )


def check_presence(item: ChecklistItem, seg: SegmentationResult) -> ChecklistItemResult:
    expected = {"present": True}
    comps, reason = _usable(item, seg, item.label)
    measured = {"count": len(comps)}
    ev = Evidence(tool="measure_count", components=comps)
    if comps:
        return _result(item, CheckStatus.PASS, measured, expected, ev,
                       f"'{item.label}' is present ({len(comps)} found).", None)
    if seg.components:
        return _result(item, CheckStatus.FAIL, measured, expected, ev,
                       f"'{item.label}' is missing.", None)
    return _result(item, CheckStatus.UNSURE, measured, expected, ev,
                   "Nothing could be segmented in this image.", reason)


def check_position(item: ChecklistItem, seg: SegmentationResult) -> ChecklistItemResult:
    assert item.relation and item.reference_label
    expected = {"relation": item.relation, "reference_label": item.reference_label}
    targets, t_reason = _usable(item, seg, item.label)
    refs, r_reason = _usable(item, seg, item.reference_label)
    ev = Evidence(tool="measure_position", components=targets + refs)
    measured: dict[str, Any] = {
        "target_count": len(targets),
        "reference_count": len(refs),
    }
    if not targets or not refs:
        reason = t_reason or r_reason or "target or reference part was not segmented"
        return _result(item, CheckStatus.UNSURE, measured, expected, ev,
                       "Cannot judge position without both parts visible.", reason)
    t, r = targets[0], refs[0]
    assert t.centroid and r.centroid
    tc, rc = t.centroid, r.centroid
    measured["target_centroid"] = [round(tc[0], 1), round(tc[1], 1)]
    measured["reference_centroid"] = [round(rc[0], 1), round(rc[1], 1)]
    holds = {
        "left_of": tc[0] < rc[0],
        "right_of": tc[0] > rc[0],
        "above": tc[1] < rc[1],
        "below": tc[1] > rc[1],
    }[item.relation]
    status = CheckStatus.PASS if holds else CheckStatus.FAIL
    text = (
        f"'{item.label}' is {item.relation.replace('_', ' ')} '{item.reference_label}' "
        f"as required."
        if holds
        else f"'{item.label}' is not {item.relation.replace('_', ' ')} "
        f"'{item.reference_label}'."
    )
    return _result(item, status, measured, expected, ev, text, None)


def check_area_ratio(item: ChecklistItem, seg: SegmentationResult) -> ChecklistItemResult:
    expected: dict[str, Any] = {"min_ratio": item.min_ratio, "max_ratio": item.max_ratio}
    targets, t_reason = _usable(item, seg, item.label)
    ev = Evidence(tool="measure_area_ratio", components=list(targets))
    if item.reference_label:
        refs, r_reason = _usable(item, seg, item.reference_label)
        ev.components = list(targets) + list(refs)
        if not targets or not refs:
            return _result(item, CheckStatus.UNSURE, {"ratio": None}, expected, ev,
                           "Cannot compute an area ratio without both parts visible.",
                           t_reason or r_reason)
        ref_area = sum(c.area_px for c in refs)
        ratio = (sum(c.area_px for c in targets) / ref_area) if ref_area else None
        base = f"vs '{item.reference_label}'"
    else:
        if not targets:
            return _result(item, CheckStatus.UNSURE, {"ratio": None}, expected, ev,
                           "Target part was not segmented confidently.", t_reason)
        ratio = sum(c.area_px for c in targets) / float(seg.image_area_px)
        base = "of the image"
    measured = {"ratio": round(float(ratio), 4) if ratio is not None else None}
    if ratio is None:
        return _result(item, CheckStatus.UNSURE, measured, expected, ev,
                       "Reference area is zero.", "reference mask is empty")
    ok = True
    if item.min_ratio is not None and ratio < item.min_ratio:
        ok = False
    if item.max_ratio is not None and ratio > item.max_ratio:
        ok = False
    status = CheckStatus.PASS if ok else CheckStatus.FAIL
    text = (
        f"'{item.label}' covers ratio {ratio:.3f} {base}, inside the spec range."
        if ok
        else f"'{item.label}' covers ratio {ratio:.3f} {base}, outside the spec range "
        f"[{item.min_ratio}, {item.max_ratio}]."
    )
    return _result(item, status, measured, expected, ev, text, None)


def check_alignment(item: ChecklistItem, seg: SegmentationResult) -> ChecklistItemResult:
    assert item.axis
    max_spread = item.max_spread_px if item.max_spread_px is not None else float("inf")
    expected = {"axis": item.axis, "max_spread_px": item.max_spread_px}
    comps, reason = _usable(item, seg, item.label)
    ev = Evidence(tool="measure_alignment", components=comps)
    if len(comps) < 2:
        return _result(item, CheckStatus.UNSURE, {"spread_px": None, "count": len(comps)},
                       expected, ev,
                       "Alignment needs at least two visible parts.", reason)
    idx = 1 if item.axis == "y" else 0
    coords = [c.centroid[idx] for c in comps if c.centroid]
    spread = float(max(coords) - min(coords)) if coords else None
    measured = {"spread_px": round(spread, 2) if spread is not None else None,
                "count": len(comps)}
    if spread is None:
        return _result(item, CheckStatus.UNSURE, measured, expected, ev,
                       "Centroids unavailable.", "masks have no area")
    ok = spread <= max_spread
    status = CheckStatus.PASS if ok else CheckStatus.FAIL
    text = (
        f"'{item.label}' parts are aligned along {item.axis} (spread {spread:.1f}px)."
        if ok
        else f"'{item.label}' parts are misaligned along {item.axis}: "
        f"spread {spread:.1f}px exceeds {max_spread:.1f}px."
    )
    return _result(item, status, measured, expected, ev, text, None)


_CHECKERS = {
    "count": check_count,
    "presence": check_presence,
    "position": check_position,
    "area_ratio": check_area_ratio,
    "alignment": check_alignment,
}


def run_check(item: ChecklistItem, seg: SegmentationResult) -> ChecklistItemResult:
    return _CHECKERS[item.check](item, seg)
