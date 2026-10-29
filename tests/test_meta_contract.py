"""Contract for the unwired Meta adapters: fail loudly, never fake output."""

import numpy as np
import pytest

from speccheck.adapters.meta import (
    MetaSamSegmenter,
    MetaSparkReasoner,
    MetaVoiceTranscriber,
)


def test_meta_adapters_require_key(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="META_API_KEY"):
        MetaSamSegmenter().segment(np.zeros((8, 8, 3), dtype=np.uint8), ["screw"])


def test_meta_adapters_are_explicit_todos(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key-not-real")
    with pytest.raises(NotImplementedError, match="TODO"):
        MetaSamSegmenter().segment(np.zeros((8, 8, 3), dtype=np.uint8), ["screw"])
    with pytest.raises(NotImplementedError, match="TODO"):
        MetaSparkReasoner().explain([], None)
    with pytest.raises(NotImplementedError, match="TODO"):
        MetaVoiceTranscriber().transcribe(b"audio")


def test_build_adapters_meta_mode_constructs(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key-not-real")
    from speccheck.adapters import build_adapters

    segmenter, reasoner, transcriber = build_adapters("meta")
    assert segmenter.name.startswith("meta")
    assert reasoner.name.startswith("meta")
    assert transcriber.name.startswith("meta")
