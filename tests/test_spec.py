import pytest

from speccheck.samples import DEMO_SPEC_YAML
from speccheck.spec import load_checklist_yaml


def test_demo_spec_parses():
    spec = load_checklist_yaml(DEMO_SPEC_YAML)
    assert spec.name == "demo-board"
    assert [i.id for i in spec.items] == [
        "screws_count",
        "led_left_of_button",
        "chip_present",
        "screws_aligned",
    ]
    assert set(spec.prompts()) == {"screw", "led", "button", "chip"}


def test_prompt_defaults_to_label():
    spec = load_checklist_yaml(
        "name: x\nitems:\n  - id: a\n    label: resistor\n    check: presence\n"
    )
    assert spec.items[0].prompt == "resistor"


def test_invalid_check_type_rejected():
    with pytest.raises(ValueError, match="check"):
        load_checklist_yaml(
            "name: x\nitems:\n  - id: a\n    label: r\n    check: magic\n"
        )


def test_position_requires_relation_and_reference():
    with pytest.raises(ValueError, match="relation"):
        load_checklist_yaml(
            "name: x\nitems:\n  - id: a\n    label: led\n    check: position\n"
        )


def test_empty_items_rejected():
    with pytest.raises(ValueError, match="items"):
        load_checklist_yaml("name: x\nitems: []\n")


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        load_checklist_yaml(
            "name: x\nitems:\n"
            "  - id: a\n    label: r\n    check: presence\n"
            "  - id: a\n    label: r\n    check: presence\n"
        )
