"""Score SpecCheck on the eval set. Run after generate_eval.py:

    uv run python eval/generate_eval.py
    uv run python eval/run_eval.py

Reports exact-match accuracy per checklist item, plus UNSURE behaviour:
how often expected-UNSURE items are correctly called UNSURE, how often
UNSURE is used on decidable items, and any false PASS (the unsafe error).
Writes eval/results.json. Only numbers this script prints belong in docs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from speccheck.adapters.mock import MockReasoner, MockSegmenter  # noqa: E402
from speccheck.agent import run_check  # noqa: E402
from speccheck.spec import load_checklist_yaml  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent


def main() -> int:
    expected_path = EVAL_DIR / "expected.json"
    if not expected_path.exists():
        print("expected.json missing - run: uv run python eval/generate_eval.py")
        return 1
    expected: dict[str, dict[str, str]] = json.loads(expected_path.read_text())
    spec = load_checklist_yaml(
        (Path(__file__).resolve().parent.parent / "samples" / "demo_spec.yaml").read_text()
    )
    segmenter, reasoner = MockSegmenter(), MockReasoner()

    rows = []
    total = correct = 0
    exp_unsure = exp_unsure_correct = 0
    pred_unsure = pred_unsure_correct = 0
    false_pass = 0
    images_all_correct = 0

    for name in sorted(expected):
        image_path = EVAL_DIR / "images" / f"{name}.png"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        report = run_check(image, spec, segmenter, reasoner, attempt=1)
        predicted = {r.item_id: r.status.value for r in report.results}
        image_ok = True
        for item_id, exp in sorted(expected[name].items()):
            pred = predicted.get(item_id, "MISSING")
            ok = pred == exp
            total += 1
            correct += int(ok)
            image_ok = image_ok and ok
            if exp == "UNSURE":
                exp_unsure += 1
                exp_unsure_correct += int(pred == "UNSURE")
            if pred == "UNSURE":
                pred_unsure += 1
                pred_unsure_correct += int(exp == "UNSURE")
            if pred == "PASS" and exp != "PASS":
                false_pass += 1
            rows.append(
                {"image": name, "item": item_id, "expected": exp, "predicted": pred, "correct": ok}
            )
        images_all_correct += int(image_ok)

    accuracy = correct / total if total else 0.0
    unsure_recall = exp_unsure_correct / exp_unsure if exp_unsure else None
    unsure_precision = pred_unsure_correct / pred_unsure if pred_unsure else None
    summary = {
        "adapter": "mock",
        "images": len(expected),
        "items_scored": total,
        "item_accuracy": round(accuracy, 4),
        "images_fully_correct": images_all_correct,
        "expected_unsure_items": exp_unsure,
        "unsure_recall": round(unsure_recall, 4) if unsure_recall is not None else None,
        "predicted_unsure_items": pred_unsure,
        "unsure_precision": round(unsure_precision, 4) if unsure_precision is not None else None,
        "false_pass_count": false_pass,
    }
    (EVAL_DIR / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n"
    )
    for row in rows:
        mark = "ok " if row["correct"] else "MISS"
        print(f"{mark} {row['image']:<22} {row['item']:<20} expected={row['expected']:<6} predicted={row['predicted']}")
    print()
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("wrote eval/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
