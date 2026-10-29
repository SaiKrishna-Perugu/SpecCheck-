"""Generate the 10-image eval set + expected per-item outcomes.

Run: ``uv run python eval/generate_eval.py``
Images are synthetic (flat-colour parts) so the mock segmenter can score
them. Expected statuses follow the demo spec in samples/demo_spec.yaml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from speccheck.samples import draw_parts, sample_blank, sample_pass  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
IMAGES_DIR = EVAL_DIR / "images"

SCREWS_ROW = [
    ("screw", 20, 20, 18, 18),
    ("screw", 70, 20, 18, 18),
    ("screw", 120, 20, 18, 18),
    ("screw", 170, 20, 18, 18),
]
LED_BUTTON_CHIP = [
    ("led", 30, 100, 22, 22),
    ("button", 90, 100, 26, 26),
    ("chip", 180, 90, 60, 44),
]

# name -> (image array, {item_id: expected status})
CASES = {
    "01_pass": (sample_pass(), {"screws_count": "PASS", "led_left_of_button": "PASS", "chip_present": "PASS", "screws_aligned": "PASS"}),
    "02_missing_screw": (
        draw_parts(SCREWS_ROW[:3] + LED_BUTTON_CHIP),
        {"screws_count": "FAIL", "led_left_of_button": "PASS", "chip_present": "PASS", "screws_aligned": "PASS"},
    ),
    "03_extra_screw": (
        draw_parts(SCREWS_ROW + [("screw", 220, 20, 18, 18)] + LED_BUTTON_CHIP),
        {"screws_count": "FAIL", "led_left_of_button": "PASS", "chip_present": "PASS", "screws_aligned": "PASS"},
    ),
    "04_misaligned_screw": (
        draw_parts(SCREWS_ROW[:3] + [("screw", 170, 80, 18, 18)] + LED_BUTTON_CHIP),
        {"screws_count": "PASS", "led_left_of_button": "PASS", "chip_present": "PASS", "screws_aligned": "FAIL"},
    ),
    "05_blank_frame": (
        sample_blank(),
        {"screws_count": "UNSURE", "led_left_of_button": "UNSURE", "chip_present": "UNSURE", "screws_aligned": "UNSURE"},
    ),
    "06_led_wrong_side": (
        draw_parts(SCREWS_ROW + [("button", 30, 100, 26, 26), ("led", 90, 100, 22, 22), ("chip", 180, 90, 60, 44)]),
        {"screws_count": "PASS", "led_left_of_button": "FAIL", "chip_present": "PASS", "screws_aligned": "PASS"},
    ),
    "07_chip_missing": (
        draw_parts(SCREWS_ROW + [("led", 30, 100, 22, 22), ("button", 90, 100, 26, 26)]),
        {"screws_count": "PASS", "led_left_of_button": "PASS", "chip_present": "FAIL", "screws_aligned": "PASS"},
    ),
    "08_tiny_screws": (
        draw_parts(
            [("screw", 20, 20, 5, 5), ("screw", 70, 20, 5, 5),
             ("screw", 120, 20, 5, 5), ("screw", 170, 20, 5, 5)] + LED_BUTTON_CHIP
        ),
        {"screws_count": "UNSURE", "led_left_of_button": "PASS", "chip_present": "PASS", "screws_aligned": "UNSURE"},
    ),
    "09_no_screws": (
        draw_parts(LED_BUTTON_CHIP),
        {"screws_count": "FAIL", "led_left_of_button": "PASS", "chip_present": "PASS", "screws_aligned": "UNSURE"},
    ),
    "10_only_screws": (
        draw_parts(SCREWS_ROW),
        {"screws_count": "PASS", "led_left_of_button": "UNSURE", "chip_present": "FAIL", "screws_aligned": "PASS"},
    ),
}


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    expected = {}
    for name, (image, outcomes) in CASES.items():
        Image.fromarray(image).save(IMAGES_DIR / f"{name}.png")
        expected[name] = outcomes
    (EVAL_DIR / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {len(CASES)} images to {IMAGES_DIR} and expected.json")


if __name__ == "__main__":
    main()
