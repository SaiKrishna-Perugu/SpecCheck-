import numpy as np

from speccheck import samples
from speccheck.adapters.mock import MockSegmenter
from speccheck.spec import load_checklist_yaml
from speccheck.tools import (
    build_report,
    compare_to_spec,
    measure_count,
    measure_position,
    request_recapture,
    segment,
)
from speccheck.types import CheckStatus, mask_from_rle, mask_to_rle


def _seg_and_spec(image):
    spec = load_checklist_yaml(samples.DEMO_SPEC_YAML)
    return segment(image, spec, MockSegmenter()), spec


def test_measure_count_and_position():
    seg, _ = _seg_and_spec(samples.sample_pass())
    assert measure_count(seg, "screw") == 4
    assert measure_count(seg, "capacitor") == 0
    pos = measure_position(seg, "led", "button", "left_of")
    assert pos["status"] == "PASS"


def test_compare_to_spec_covers_every_item():
    seg, spec = _seg_and_spec(samples.sample_pass())
    results = compare_to_spec(spec, seg)
    assert len(results) == len(spec.items)
    assert all(r.status == CheckStatus.PASS for r in results)


def test_request_recapture_lifecycle():
    seg, spec = _seg_and_spec(samples.sample_blank())
    results = compare_to_spec(spec, seg)
    req = request_recapture(results, attempt=1)
    assert req is not None and req.retries_left == 2 and req.reasons
    assert request_recapture(results, attempt=3) is None  # exhausted
    seg2, spec2 = _seg_and_spec(samples.sample_pass())
    assert request_recapture(compare_to_spec(spec2, seg2), attempt=1) is None


def test_mask_rle_roundtrip():
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:5, 3:7] = True
    assert np.array_equal(mask_from_rle(mask_to_rle(mask)), mask)


def test_build_report_links_evidence_and_renders():
    seg, spec = _seg_and_spec(samples.sample_pass())
    results = compare_to_spec(spec, seg)
    report = build_report(
        spec_name=spec.name, results=results, image=samples.sample_pass(), adapter="mock"
    )
    data = report.to_dict()
    assert data["summary"] == {"PASS": 4, "FAIL": 0, "UNSURE": 0}
    assert data["annotated_image_png_base64"]
    for item in data["items"]:
        assert item["evidence"]["components"], item["item_id"]
        assert item["evidence"]["tool"]
    html = report.to_html()
    assert "screws_count" in html and "PASS" in html
