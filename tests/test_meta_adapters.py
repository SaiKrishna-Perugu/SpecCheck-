"""Meta adapter behaviour with a fake transport (no network, no key use)."""

import numpy as np
import pytest

import meta_sam_parser
from speccheck.adapters import meta
from speccheck.adapters.meta import (
    MetaSamSegmenter,
    MetaSparkReasoner,
    MetaVoiceTranscriber,
    parse_sam_output,
)
from speccheck.types import ChecklistItemResult, CheckStatus, Evidence


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key-not-real")


def test_sam_segmenter_parses_records(monkeypatch):
    monkeypatch.setattr(
        meta_sam_parser,
        "decode_mask_to_raster",
        lambda mask: bytes([1]) * (mask.width * mask.height),
    )

    def fake_post(path, payload):
        assert path == "/responses"
        assert payload["model"] == "sam-3.1"
        assert payload["metadata"] == {
            "include_confidence": "true",
            "mask_encoding": "one_bit",
        }
        text = (
            "<0f>0<|box;x1=2;y1=3;x2=5;y2=6;w=20;h=20;c=0.9|>"
            "<|mask;x=0;y=0;c=0.9;data=4,4,!payload|>"
        )
        return {"output": [{"content": [{"type": "output_text", "text": text}]}]}

    monkeypatch.setattr(meta, "_post_json", fake_post)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    result = MetaSamSegmenter().segment(image, ["screw"])
    assert len(result.components) == 1
    comp = result.components[0]
    assert comp.label == "screw"
    assert comp.confidence == 0.9
    assert comp.area_px == 16
    assert comp.mask[3, 2]


def test_sam_zero_matches_is_empty_not_error(monkeypatch):
    monkeypatch.setattr(meta, "_post_json", lambda path, payload: {"output_text": ""})
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert MetaSamSegmenter().segment(image, ["unicorn"]).components == []


def test_spark_explainer_uses_spark_model(monkeypatch):
    seen = {}

    def fake_post(path, payload):
        seen.update(payload)
        return {"output_text": "All four screws are present and aligned."}

    monkeypatch.setattr(meta, "_post_json", fake_post)
    result = ChecklistItemResult(
        item_id="screws_count", label="screw", check="count",
        status=CheckStatus.PASS, measured={"count": 4},
        expected={"expected_count": 4}, evidence=Evidence(tool="measure_count"),
        explanation="Found 4.",
    )
    text = MetaSparkReasoner().explain([result], "check it")
    assert seen["model"] == "muse-spark-1.3"
    assert "screws" in text.lower()


def test_voice_transcriber_parses_transcript(monkeypatch):
    seen = {}

    def fake_multipart(path, fields):
        seen["path"] = path
        seen["fields"] = fields
        return {"transcript": "check the screws", "audioDurationMs": 900}

    monkeypatch.setattr(meta, "_post_multipart", fake_multipart)
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    result = MetaVoiceTranscriber().transcribe(buf.getvalue(), filename="a.wav")
    assert result.text == "check the screws"
    assert seen["path"] == "/asr/transcribe"
    assert b"muse-voice-transcribe-1.0" in seen["fields"]["request"][1]


def test_parse_real_wire_payload_decodes():
    # Captured live from sam-3.1 (one_bit mask of an 18x18 square).
    text = (
        "<0f>0<|box;x1=20;y1=20;x2=37;y2=37;w=320;h=240;c=0.934|>"
        "<|mask;x=0;y=0;c=0.934;data=18,18,!!!!!'y:OOxq7|>"
    )
    comps = parse_sam_output(text, image_width=320, image_height=240, prompt="gray square")
    assert len(comps) == 1
    assert comps[0].area_px == 324
    assert comps[0].confidence == 0.934
    assert comps[0].bbox.x_min == 20


def test_parse_sam_output_skips_malformed_records(monkeypatch):
    monkeypatch.setattr(
        meta_sam_parser, "decode_mask_to_raster", lambda mask: bytes(mask.width * mask.height)
    )
    assert parse_sam_output("garbage", image_width=10, image_height=10, prompt="x") == []
