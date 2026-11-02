"""Generate the real-style eval set: screenshots of the 3D board renderer.

These are not flat clip-art: perspective, shading, shadows, and labels make
them closer to photos than the synthetic set, while expected verdicts stay
known. Run with a Playwright-capable python (see docs in README):

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
CASES = {
    "r01_pass": ("Pass", dict(ALL_PASS)),
    "r02_missing": ("Missing screw", {**ALL_PASS, "screws_count": "FAIL"}),
    "r03_misaligned": ("Misaligned", {**ALL_PASS, "screws_aligned": "FAIL"}),
    "r04_blank": (
        "Blank frame",
        {k: "UNSURE" for k in ALL_PASS},
    ),
}


def main() -> None:
    from playwright.sync_api import sync_playwright

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    expected = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROME, headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 760})
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_timeout(2500)
        for name, (label, outcomes) in CASES.items():
            page.get_by_role("button", name=label, exact=True).click()
            page.wait_for_timeout(1200)
            page.locator("#scene").screenshot(path=str(IMAGES_DIR / f"{name}.png"))
            expected[name] = outcomes
        browser.close()
    (REAL_DIR / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {len(CASES)} real-style images to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
