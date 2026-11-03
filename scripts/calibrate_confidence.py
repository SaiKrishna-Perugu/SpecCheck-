"""Calibrate SAM confidence against correctness on the real-photo set.

For every labelled photo, segments each spec prompt once and records the
top confidence per label next to whether the presence verdict at several
candidate thresholds would be right. Prints accuracy/false-PASS per
threshold so min_confidence is chosen from data, not defaults.

Run: ``uv run python scripts/calibrate_confidence.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from speccheck.adapters.meta import MetaSamSegmenter  # noqa: E402
from speccheck.config import get_settings  # noqa: E402
from speccheck.spec import load_checklist_yaml  # noqa: E402

BASE = Path(__file__).resolve().parent.parent / "eval" / "realphotos"
THRESHOLDS = [0.5, 0.65, 0.75, 0.85, 0.92]


def main() -> int:
    get_settings()
    expected = json.loads((BASE / "expected.json").read_text())
    spec = load_checklist_yaml((BASE / "spec.yaml").read_text())
    segmenter = MetaSamSegmenter()
    rows = []
    for name in sorted(expected):
        path = BASE / "images" / f"{name}.jpg"
        image = np.asarray(Image.open(path).convert("RGB"))
        seg = segmenter.segment(image, spec.prompts(), spec.color_hints())
        mapping = spec.prompt_to_label()
        top: dict[str, float] = {}
        for comp in seg.components:
            label = mapping.get(comp.prompt, comp.label)
            top[label] = max(top.get(label, 0.0), comp.confidence)
        for item in spec.items:
            rows.append((name, item.label, top.get(item.label), expected[name][item.id]))
            print(f"{name:<12} {item.label:<6} top_conf={top.get(item.label)} expected={expected[name][item.id]}")
    print()
    for thr in THRESHOLDS:
        correct = false_pass = 0
        for _, label, conf, exp in rows:
            present = conf is not None and conf >= thr
            pred = "PASS" if present else ("UNSURE" if conf is not None else "FAIL")
            # With no mask at all the pipeline says FAIL only when other
            # labels segmented; approximate with FAIL here for calibration.
            correct += int(pred == exp)
            false_pass += int(pred == "PASS" and exp != "PASS")
        print(f"threshold {thr:.2f}: accuracy {correct}/{len(rows)} = {correct/len(rows):.4f}, false_pass {false_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
