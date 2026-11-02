"""Baseline: Muse Spark alone judges each eval image (no deterministic checks).

Scores VLM-only verdicts with the same expected outcomes as eval/run_eval.py
so the deterministic pipeline and the VLM can be compared honestly, false
PASSes included. Costs real API calls; requires SPECCHECK_ADAPTER env or
.env with META_API_KEY.

Run: ``uv run python scripts/baseline_vlm.py [--hard]``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from speccheck import samples  # noqa: E402
from speccheck.adapters import meta  # noqa: E402
from speccheck.config import get_settings  # noqa: E402
from speccheck.spec import load_checklist_yaml  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def ask_vlm(image_path: Path, spec_yaml: str) -> dict[str, str]:
    import numpy as np

    image = np.asarray(Image.open(image_path).convert("RGB"))
    prompt = (
        "You are judging an assembly photo against this checklist (YAML):\n"
        f"{spec_yaml}\n"
        "For each checklist item id, answer PASS, FAIL, or UNSURE (UNSURE when "
        "you cannot see enough evidence). Reply with ONLY a JSON object mapping "
        "item id to verdict, no other text."
    )
    response = meta._post_json(
        "/responses",
        {
            "model": meta.SPARK_MODEL,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": meta._image_data_url(image)},
                    ],
                }
            ],
            "stream": False,
        },
    )
    text = meta._output_text(response).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v).upper() for k, v in data.items()}


def main() -> int:
    get_settings()
    hard = "--hard" in sys.argv
    dataset_dir = EVAL_DIR / "hard" if hard else EVAL_DIR
    expected: dict[str, dict[str, str]] = json.loads(
        (dataset_dir / "expected.json").read_text()
    )
    total = correct = false_pass = 0
    exp_unsure = exp_unsure_correct = 0
    for name in sorted(expected):
        predicted = ask_vlm(dataset_dir / "images" / f"{name}.png", samples.DEMO_SPEC_YAML)
        for item_id, exp in sorted(expected[name].items()):
            pred = predicted.get(item_id, "UNSURE")
            total += 1
            correct += int(pred == exp)
            false_pass += int(pred == "PASS" and exp != "PASS")
            if exp == "UNSURE":
                exp_unsure += 1
                exp_unsure_correct += int(pred == "UNSURE")
            print(f"{name:<22} {item_id:<20} expected={exp:<6} vlm={pred}")
    print()
    print(f"dataset: {'hard' if hard else 'standard'} (VLM-only, {meta.SPARK_MODEL})")
    print(f"items_scored: {total}")
    print(f"item_accuracy: {round(correct / total, 4) if total else 0}")
    print(f"false_pass_count: {false_pass}")
    print(
        f"unsure_recall: {round(exp_unsure_correct / exp_unsure, 4) if exp_unsure else None}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
