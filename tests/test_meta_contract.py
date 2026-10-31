"""Contract for the Meta adapters: key required, no silent fakes."""

import numpy as np
import pytest

from speccheck.adapters import meta
from speccheck.adapters.meta import (
    MetaSamSegmenter,
    MetaSparkReasoner,
    MetaVoiceTranscriber,
)


def test_meta_adapters_require_key(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setattr("speccheck.config._load_dotenv", lambda path: None)
    with pytest.raises(RuntimeError, match="META_API_KEY"):
        MetaSamSegmenter().segment(np.zeros((8, 8, 3), dtype=np.uint8), ["screw"])
    with pytest.raises(RuntimeError, match="META_API_KEY"):
        MetaSparkReasoner().explain([], None)
    with pytest.raises(RuntimeError, match="META_API_KEY"):
        MetaVoiceTranscriber().transcribe(b"audio")


def test_build_adapters_meta_mode_constructs():
    from speccheck.adapters import build_adapters

    segmenter, reasoner, transcriber = build_adapters("meta")
    assert segmenter.name == "meta-sam-3.1"
    assert reasoner.name == "meta-muse-spark-1.3"
    assert transcriber.name == "meta-muse-voice-transcribe-1.0"


def test_transcriber_rejects_non_wav_without_ffmpeg(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key-not-real")
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(ValueError, match="WAV"):
        MetaVoiceTranscriber().transcribe(b"not-audio", filename="clip.webm")
