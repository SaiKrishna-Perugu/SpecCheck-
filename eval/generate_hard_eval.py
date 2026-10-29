"""Generate the hard eval set: degraded / occluded variants of the demo board.

Run: ``uv run python eval/generate_hard_eval.py``
Expected statuses are ground-truth QA verdicts, not whatever the current
pipeline happens to output - misses on this set are real findings.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from speccheck.samples import BACKGROUND, draw_parts  # noqa: E402

HARD_DIR = Path(__file__).resolve().parent / "hard"
IMAGES_DIR = HARD_DIR / "images"
rng = np.random.default_rng(7)

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
ALL_PARTS = SCREWS_ROW + LED_BUTTON_CHIP
ALL_PASS = {
    "screws_count": "PASS",
    "led_left_of_button": "PASS",
    "chip_present": "PASS",
    "screws_aligned": "PASS",
}


def _noise(image: np.ndarray, sigma: float) -> np.ndarray:
    noisy = image.astype(float) + rng.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def _brightness(image: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(image.astype(float) * factor, 0, 255).astype(np.uint8)


def _blur(image: np.ndarray, radius: float) -> np.ndarray:
    return np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius)))


def _rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    pil = Image.fromarray(image).rotate(
        degrees, resample=Image.BILINEAR, fillcolor=BACKGROUND
    )
    return np.asarray(pil)


def _occlude(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    out = image.copy()
    out[y : y + h, x : x + w] = BACKGROUND
    return out


BASE = draw_parts(ALL_PARTS)

CASES = {
    "h01_noise": (_noise(BASE, 18), dict(ALL_PASS)),
    "h02_jpeg_low": (_jpeg(BASE, 35), dict(ALL_PASS)),
    "h03_dark": (_brightness(BASE, 0.55), dict(ALL_PASS)),
    "h04_bright": (_brightness(BASE, 1.45), dict(ALL_PASS)),
    "h05_blur": (_blur(BASE, 1.6), dict(ALL_PASS)),
    "h06_noise_missing_screw": (
        _noise(draw_parts(SCREWS_ROW[:3] + LED_BUTTON_CHIP), 18),
        {**ALL_PASS, "screws_count": "FAIL"},
    ),
    "h07_jpeg_extra_screw": (
        _jpeg(draw_parts(SCREWS_ROW + [("screw", 220, 20, 18, 18)] + LED_BUTTON_CHIP), 40),
        {**ALL_PASS, "screws_count": "FAIL"},
    ),
    "h08_chip_half_occluded": (
        _occlude(BASE, 180, 90, 30, 44),
        dict(ALL_PASS),  # chip still visible: presence must still pass
    ),
    "h09_chip_fully_occluded": (
        _occlude(BASE, 175, 85, 70, 54),
        # Ground truth: chip exists but is hidden, so UNSURE is correct.
        # Indistinguishable from missing with current checks - a known gap.
        {**ALL_PASS, "chip_present": "UNSURE"},
    ),
    "h10_rotated_board": (
        _rotate(BASE, 12),
        # Rotation preserves left/right order and counts, but image-axis
        # alignment of the screw row no longer holds.
        {**ALL_PASS, "screws_aligned": "FAIL"},
    ),
    "h11_noise_and_blur": (_blur(_noise(BASE, 14), 1.2), dict(ALL_PASS)),
    "h12_dark_jpeg_missing_chip": (
        _jpeg(_brightness(draw_parts(SCREWS_ROW + LED_BUTTON_CHIP[:2]), 0.6), 40),
        {**ALL_PASS, "chip_present": "FAIL"},
    ),
}


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    expected = {}
    for name, (image, outcomes) in CASES.items():
        Image.fromarray(image).save(IMAGES_DIR / f"{name}.png")
        expected[name] = outcomes
    (HARD_DIR / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {len(CASES)} hard images to {IMAGES_DIR} and hard/expected.json")


if __name__ == "__main__":
    main()
