from speccheck import samples
from speccheck.adapters.mock import MockReasoner, MockSegmenter, MockTranscriber
from speccheck.agent import run_check
from speccheck.spec import load_checklist_yaml
from speccheck.types import CheckStatus


def _run(image, attempt=1, segmenter=None):
    spec = load_checklist_yaml(samples.DEMO_SPEC_YAML)
    return run_check(
        image,
        spec,
        segmenter or MockSegmenter(),
        MockReasoner(),
        transcript="check the assembly against the spec",
        attempt=attempt,
    )


def test_pass_report_all_pass_no_recapture():
    report = _run(samples.sample_pass())
    assert report.summary() == {"PASS": 4, "FAIL": 0, "UNSURE": 0}
    assert report.needs_recapture is False


def test_missing_screw_fails_without_recapture():
    report = _run(samples.sample_missing())
    by_id = {r.item_id: r for r in report.results}
    assert by_id["screws_count"].status == CheckStatus.FAIL
    assert report.needs_recapture is False


def test_blank_asks_recapture_then_gives_up_after_max_attempts():
    first = _run(samples.sample_blank(), attempt=1)
    assert first.needs_recapture is True
    assert first.recapture_reasons
    assert all(r.status == CheckStatus.UNSURE for r in first.results)
    last = _run(samples.sample_blank(), attempt=3)
    assert last.needs_recapture is False
    assert all(r.status == CheckStatus.UNSURE for r in last.results)


def test_low_confidence_requests_recapture():
    report = _run(samples.sample_pass(), segmenter=MockSegmenter(confidence=0.2))
    assert report.needs_recapture is True


def test_mock_transcriber_is_honest_about_being_mock():
    t = MockTranscriber()
    assert t.transcribe(b"").text == ""
    result = t.transcribe(b"fake-audio-bytes")
    assert result.adapter.startswith("mock")
    assert result.text  # canned command, labelled mock via adapter name
