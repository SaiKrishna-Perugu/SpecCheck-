import numpy as np

from speccheck import samples
from speccheck.adapters.fallback import ClassicalSegmenter
from speccheck.spec import load_checklist_yaml
from speccheck.tools import segment

# Non-palette photo-like board: crimson + teal parts on a beige bench.
PHOTO_SPEC = """\
name: bench-photo
items:
  - id: red_knobs
    label: knob
    prompt: knob
    color: red
    check: count
    expected_count: 2
  - id: teal_block_present
    label: block
    prompt: block
    color: cyan
    check: presence
"""


def _photo() -> np.ndarray:
    image = np.full((240, 320, 3), (200, 190, 170), dtype=np.uint8)
    image[30:55, 30:55] = (190, 30, 25)  # crimson knob
    image[30:55, 90:115] = (195, 35, 30)  # crimson knob
    image[100:140, 60:110] = (30, 170, 180)  # teal block
    return image


def test_fallback_finds_parts_by_color_hint():
    spec = load_checklist_yaml(PHOTO_SPEC)
    seg = segment(_photo(), spec, ClassicalSegmenter())
    assert len(seg.by_label("knob")) == 2
    assert len(seg.by_label("block")) == 1
    assert all(c.confidence == 0.65 for c in seg.components)


def test_fallback_without_hints_stays_honest():
    neutral = (
        "name: neutral\nitems:\n"
        "  - id: a\n    label: knob\n    prompt: knob\n    check: presence\n"
        "  - id: b\n    label: block\n    prompt: block\n    check: presence\n"
    )
    spec = load_checklist_yaml(neutral)  # no colour hints or colour words
    seg = segment(_photo(), spec, ClassicalSegmenter())
    assert seg.components == []


def test_fallback_blank_is_empty():
    spec = load_checklist_yaml(PHOTO_SPEC)
    blank = np.full((240, 320, 3), (200, 190, 170), dtype=np.uint8)
    assert segment(blank, spec, ClassicalSegmenter()).components == []
