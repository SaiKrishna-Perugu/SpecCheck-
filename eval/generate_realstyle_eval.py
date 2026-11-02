"""Generate the photo-style eval set: screenshots of the 3D board renderer.

32 labelled images: 5 scenarios x camera angles/lighting variations of the
Three.js board. These are renders, not phone photos - perspective, shading
and shadows make them harder than clip-art, and the README says exactly
that. Expected verdicts are scenario ground truth.

Run with a Playwright-capable python:

    .tmp/video-env/Scripts/python.exe eval/generate_realstyle_eval.py
    uv run python eval/run_eval.py --real [--adapter=fallback|meta]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REAL_DIR = Path(__file__).resolve().parent / "realstyle"
IMAGES_DIR = REAL_DIR / "images"
CHROME = r"C:\Users\psaik\AppData\Local\ms-playwright\chromium-1234\chrome-win64\chrome.exe"
BASE = "http://127.0.0.1:8000/3d"

ALL_PASS = {
    "screws_count": "PASS",
    "led_left_of_button": "PASS",
    "chip_present": "PASS",
    "screws_aligned": "PASS",
}
SCENARIOS = {
    "pass": ("Pass", dict(ALL_PASS)),
    "missing": ("Missing screw", {**ALL_PASS, "screws_count": "FAIL"}),
    "extra": ("Extra screw", {**ALL_PASS, "screws_count": "FAIL"}),
    "misaligned": ("Misaligned", {**ALL_PASS, "screws_aligned": "FAIL"}),
    "blank": ("Blank frame", {k: "UNSURE" for k in ALL_PASS}),
}
# (suffix, drag from -> to) camera variations; None = default view
VIEWS = [
    ("front", None),
    ("left", (500, 360, 300, 330)),
    ("right", (500, 360, 760, 330)),
    ("top", (550, 300, 560, 520)),
    ("close", None),  # wheel zoom applied instead
    ("far", None),
    ("low", (500, 420, 520, 240)),
]


def main() -> None:
    from playwright.sync_api import sync_playwright

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    expected = {}
    idx = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROME, headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 760})
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_timeout(2500)
        for scen, (label, outcomes) in SCENARIOS.items():
            for view, drag in VIEWS:
                if scen == "blank" and view not in {"front", "left"}:
                    continue
                idx += 1
                name = f"r{idx:02d}_{scen}_{view}"
                page.get_by_role("button", name="Reset view").click()
                page.wait_for_timeout(500)
                page.get_by_role("button", name=label, exact=True).click()
                page.wait_for_timeout(700)
                if drag:
                    x0, y0, x1, y1 = drag
                    page.mouse.move(x0, y0)
                    page.mouse.down()
                    page.mouse.move(x1, y1, steps=18)
                    page.mouse.up()
                elif view == "close":
                    page.mouse.wheel(0, -900)
                elif view == "far":
                    page.mouse.wheel(0, 900)
                page.wait_for_timeout(700)
                page.locator("#scene").screenshot(path=str(IMAGES_DIR / f"{name}.png"))
                expected[name] = outcomes
        browser.close()
    (REAL_DIR / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {len(expected)} photo-style images to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
