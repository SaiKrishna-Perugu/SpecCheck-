import numpy as np

from speccheck import samples
from speccheck.adapters.mock import MockSegmenter
from speccheck.checks import run_check
from speccheck.spec import ChecklistItem
from speccheck.tools import segment
from speccheck.spec import load_checklist_yaml
from speccheck.types import CheckStatus


def _seg(image):
    spec = load_checklist_yaml(samples.DEMO_SPEC_YAML)
    return segment(image, spec, MockSegmenter())


def _item(item_id):
    spec = load_checklist_yaml(samples.DEMO_SPEC_YAML)
    return next(i for i in spec.items if i.id == item_id)


def test_count_pass():
    result = run_check(_item("screws_count"), _seg(samples.sample_pass()))
    assert result.status == CheckStatus.PASS
    assert result.measured["count"] == 4
    assert result.evidence.has_evidence


def test_count_missing_is_fail_not_unsure():
    result = run_check(_item("screws_count"), _seg(samples.sample_missing()))
    assert result.status == CheckStatus.FAIL
    assert result.measured["count"] == 3


def test_count_extra_is_fail():
    result = run_check(_item("screws_count"), _seg(samples.sample_extra()))
    assert result.status == CheckStatus.FAIL
    assert result.measured["count"] == 5


def test_blank_image_is_unsure_never_pass():
    seg = _seg(samples.sample_blank())
    for item_id in ("screws_count", "chip_present", "screws_aligned", "led_left_of_button"):
        result = run_check(_item(item_id), seg)
        assert result.status == CheckStatus.UNSURE, item_id
        assert result.recapture_reason


def test_position_pass_and_fail():
    ok = run_check(_item("led_left_of_button"), _seg(samples.sample_pass()))
    assert ok.status == CheckStatus.PASS
    flipped = ChecklistItem(
        id="x", label="button", check="position",
        relation="left_of", reference_label="led",
    )
    bad = run_check(flipped, _seg(samples.sample_pass()))
    assert bad.status == CheckStatus.FAIL


def test_alignment_misaligned_fails():
    result = run_check(_item("screws_aligned"), _seg(samples.sample_misaligned()))
    assert result.status == CheckStatus.FAIL
    assert result.measured["spread_px"] > 12


def test_alignment_line_fit_tolerates_rotation_but_not_stray():
    diagonal = samples.draw_parts(
        [
            ("screw", 20, 20, 18, 18),
            ("screw", 70, 45, 18, 18),
            ("screw", 120, 70, 18, 18),
            ("screw", 170, 95, 18, 18),
        ]
    )
    seg = _seg(diagonal)
    line_item = ChecklistItem(
        id="diag", label="screw", check="alignment", axis="line", max_spread_px=12
    )
    assert run_check(line_item, seg).status == CheckStatus.PASS
    row_item = ChecklistItem(
        id="row", label="screw", check="alignment", axis="y", max_spread_px=12
    )
    assert run_check(row_item, seg).status == CheckStatus.FAIL
    stray = run_check(line_item, _seg(samples.sample_misaligned()))
    assert stray.status == CheckStatus.FAIL


def test_occluded_part_is_unsure_not_fail():
    # Huge button blob (>25% of frame) and no chip: chip may be hidden.
    image = samples.draw_parts([("button", 10, 10, 170, 130)])
    seg = _seg(image)
    result = run_check(_item("chip_present"), seg)
    assert result.status == CheckStatus.UNSURE
    assert "occlud" in (result.recapture_reason or "")


def test_area_ratio_bounds():
    item = ChecklistItem(
        id="chip_area", label="chip", check="area_ratio",
        reference_label="led", min_ratio=1.0, max_ratio=10.0,
    )
    result = run_check(item, _seg(samples.sample_pass()))
    assert result.status == CheckStatus.PASS
    assert result.measured["ratio"] > 1.0
    strict = ChecklistItem(
        id="chip_area2", label="chip", check="area_ratio",
        reference_label="led", min_ratio=50.0,
    )
    assert run_check(strict, _seg(samples.sample_pass())).status == CheckStatus.FAIL


def test_low_confidence_is_unsure_with_reason():
    spec = load_checklist_yaml(samples.DEMO_SPEC_YAML)
    seg = segment(samples.sample_pass(), spec, MockSegmenter(confidence=0.2))
    result = run_check(_item("screws_count"), seg)
    assert result.status == CheckStatus.UNSURE
    assert "confidence" in (result.recapture_reason or "")
