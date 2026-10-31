"""M1 smoke script: call each model adapter once and print its output.

With SPECCHECK_ADAPTER=mock (default) this runs fully offline. With
SPECCHECK_ADAPTER=meta it stops at the TODO stubs in
speccheck/adapters/meta.py - the real model IDs, endpoints, and payload
shapes must come from DOCS_NOTES.md first. Nothing is invented here.

Run: ``uv run python scripts/meta_smoke.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speccheck import samples  # noqa: E402
from speccheck.adapters import build_adapters  # noqa: E402
from speccheck.config import get_settings  # noqa: E402
from speccheck.spec import load_checklist_yaml  # noqa: E402
from speccheck.tools import compare_to_spec, segment  # noqa: E402


def main() -> int:
    settings = get_settings()
    print(f"adapter mode: {settings.adapter} (meta key configured: {settings.meta_configured})")
    segmenter, reasoner, transcriber = build_adapters(settings.adapter)

    spec = load_checklist_yaml(samples.DEMO_SPEC_YAML)
    image = samples.sample_pass()

    print(f"\n[segmenter: {segmenter.name}] SAM 3.1 role - text-prompt segmentation")
    seg = segment(image, spec, segmenter)
    for comp in seg.components:
        print(f"  {comp.label}: area={comp.area_px}px conf={comp.confidence} bbox={comp.bbox}")
    if settings.adapter == "meta" and not seg.components:
        # Verified live: SAM matches concrete concepts, not bare part names
        # on flat clip-art ("screw" -> zero matches; "gray square" -> all 4).
        print("  (spec prompts matched nothing on this synthetic image; trying 'gray square')")
        demo = segmenter.segment(image, ["gray square"])
        for comp in demo.components:
            print(f"  gray square: area={comp.area_px}px conf={comp.confidence:.3f} bbox={comp.bbox}")

    print(f"\n[reasoner: {reasoner.name}] Muse Spark role - plain-language explanation")
    results = compare_to_spec(spec, seg)
    print(" ", reasoner.explain(results, "check the assembly against the spec"))

    print(f"\n[transcriber: {transcriber.name}] Muse Voice Transcribe role")
    transcript = transcriber.transcribe(_silence_wav(), filename="silence.wav")
    print(f"  text={transcript.text!r} adapter={transcript.adapter}")
    return 0


def _silence_wav() -> bytes:
    """One second of valid WAV silence (the real API rejects non-WAV)."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)
    return buf.getvalue()


if __name__ == "__main__":
    raise SystemExit(main())
