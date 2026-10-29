from speccheck.intent import parse_intent
from speccheck.samples import DEMO_SPEC_YAML
from speccheck.spec import load_checklist_yaml

SPEC = load_checklist_yaml(DEMO_SPEC_YAML)


def test_full_check():
    assert parse_intent("check the assembly against the spec", SPEC).action == "full_check"


def test_check_single_item():
    intent = parse_intent("check screws count", SPEC)
    assert intent.action == "check_item"
    assert intent.item_id == "screws_count"


def test_check_item_by_label():
    intent = parse_intent("verify the chip", SPEC)
    assert intent.action in {"check_item", "full_check"}
    if intent.action == "check_item":
        assert intent.label == "chip"


def test_recount():
    intent = parse_intent("how many screws are there", SPEC)
    assert intent.action == "recount"
    assert intent.label == "screw"


def test_explain():
    intent = parse_intent("explain screws count", SPEC)
    assert intent.action == "explain"
    assert intent.item_id == "screws_count"


def test_unknown_never_guesses():
    assert parse_intent("blah blah unrelated words", SPEC).action == "unknown"
    assert parse_intent("", SPEC).action == "unknown"
    assert parse_intent(None, SPEC).action == "unknown"
