"""Synthetic sample assemblies (shared palette for mock segmenter + tests).

Images are flat-colour rectangles on a dark background so the mock segmenter
can find them deterministically. Real photos need the real SAM adapter.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

BACKGROUND = (24, 24, 28)

# Prompt keyword -> RGB used to draw that part. The mock segmenter matches
# blob colours back to prompts through this same table.
PART_COLORS: dict[str, tuple[int, int, int]] = {
    "screw": (200, 200, 205),
    "led": (230, 60, 50),
    "button": (60, 200, 90),
    "chip": (70, 110, 230),
    "board": (40, 90, 60),
    "resistor": (230, 170, 40),
    "capacitor": (180, 80, 200),
    "wire": (240, 240, 240),
    "brick": (220, 120, 40),
    "wheel": (120, 120, 130),
}

IMAGE_W, IMAGE_H = 320, 240


def blank_image() -> np.ndarray:
    return np.full((IMAGE_H, IMAGE_W, 3), BACKGROUND, dtype=np.uint8)


def draw_parts(parts: list[tuple[str, int, int, int, int]]) -> np.ndarray:
    """Draw ``(label, x, y, w, h)`` rectangles and return an RGB array."""
    img = Image.fromarray(blank_image())
    draw = ImageDraw.Draw(img)
    for label, x, y, w, h in parts:
        color = PART_COLORS[label]
        draw.rectangle([x, y, x + w - 1, y + h - 1], fill=color)
    return np.asarray(img)


def sample_pass() -> np.ndarray:
    """4 screws in a row, LED left of button, chip present."""
    parts = [
        ("screw", 20, 20, 18, 18),
        ("screw", 70, 20, 18, 18),
        ("screw", 120, 20, 18, 18),
        ("screw", 170, 20, 18, 18),
        ("led", 30, 100, 22, 22),
        ("button", 90, 100, 26, 26),
        ("chip", 180, 90, 60, 44),
    ]
    return draw_parts(parts)


def sample_missing() -> np.ndarray:
    """Only 3 screws (one missing)."""
    parts = [
        ("screw", 20, 20, 18, 18),
        ("screw", 70, 20, 18, 18),
        ("screw", 120, 20, 18, 18),
        ("led", 30, 100, 22, 22),
        ("button", 90, 100, 26, 26),
        ("chip", 180, 90, 60, 44),
    ]
    return draw_parts(parts)


def sample_extra() -> np.ndarray:
    """5 screws (one extra)."""
    return draw_parts(
        [
            ("screw", 20, 20, 18, 18),
            ("screw", 70, 20, 18, 18),
            ("screw", 120, 20, 18, 18),
            ("screw", 170, 20, 18, 18),
            ("screw", 220, 20, 18, 18),
            ("led", 30, 100, 22, 22),
            ("button", 90, 100, 26, 26),
            ("chip", 180, 90, 60, 44),
        ]
    )


def sample_misaligned() -> np.ndarray:
    """4 screws, but one sits far below the row."""
    return draw_parts(
        [
            ("screw", 20, 20, 18, 18),
            ("screw", 70, 20, 18, 18),
            ("screw", 120, 20, 18, 18),
            ("screw", 170, 80, 18, 18),
            ("led", 30, 100, 22, 22),
            ("button", 90, 100, 26, 26),
            ("chip", 180, 90, 60, 44),
        ]
    )


def sample_blank() -> np.ndarray:
    """Empty frame: nothing to segment, must come back UNSURE."""
    return blank_image()


SAMPLE_BUILDERS = {
    "pass": sample_pass,
    "missing": sample_missing,
    "extra": sample_extra,
    "misaligned": sample_misaligned,
    "blank": sample_blank,
}

DEMO_SPEC_YAML = """\
name: demo-board
items:
  - id: screws_count
    label: screw
    check: count
    expected_count: 4
  - id: led_left_of_button
    label: led
    check: position
    relation: left_of
    reference_label: button
  - id: chip_present
    label: chip
    check: presence
  - id: screws_aligned
    label: screw
    check: alignment
    axis: y
    max_spread_px: 12
"""
