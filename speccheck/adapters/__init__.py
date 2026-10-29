"""Model adapters: mock implementations now, Meta APIs behind the same interfaces."""

from .base import Reasoner, Segmenter, Transcriber, Transcript
from .mock import MockReasoner, MockSegmenter, MockTranscriber

__all__ = [
    "Segmenter",
    "Reasoner",
    "Transcriber",
    "Transcript",
    "MockSegmenter",
    "MockReasoner",
    "MockTranscriber",
    "build_adapters",
]


def build_adapters(mode: str) -> tuple[Segmenter, Reasoner, Transcriber]:
    """Return (segmenter, reasoner, transcriber) for mock / fallback / meta."""
    if mode == "fallback":
        from .fallback import ClassicalSegmenter

        return ClassicalSegmenter(), MockReasoner(), MockTranscriber()
    if mode == "meta":
        # Imported lazily: raises NotImplementedError with TODO guidance until
        # the real endpoints from DOCS_NOTES.md are wired in.
        from .meta import MetaSamSegmenter, MetaSparkReasoner, MetaVoiceTranscriber

        return MetaSamSegmenter(), MetaSparkReasoner(), MetaVoiceTranscriber()
    return MockSegmenter(), MockReasoner(), MockTranscriber()
