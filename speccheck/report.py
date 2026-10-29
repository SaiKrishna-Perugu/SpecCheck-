"""Evidence-linked report builder (HTML + JSON).

Every verdict carries its measured values, the expected values, and the
segmentation evidence (bbox / centroid / area / mask RLE) that produced it.
Items without evidence are UNSURE by construction (see speccheck.checks).
"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .types import ChecklistItemResult, CheckStatus

_STATUS_COLORS = {
    CheckStatus.PASS: (34, 160, 60),
    CheckStatus.FAIL: (210, 50, 45),
    CheckStatus.UNSURE: (220, 150, 20),
}


@dataclass
class Report:
    spec_name: str
    results: list[ChecklistItemResult]
    image_width: int
    image_height: int
    annotated_png: bytes | None = None
    transcript: str | None = None
    adapter: str = "mock"
    attempt: int = 1
    needs_recapture: bool = False
    recapture_reasons: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> dict[str, int]:
        counts = {"PASS": 0, "FAIL": 0, "UNSURE": 0}
        for r in self.results:
            counts[r.status.value] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_name": self.spec_name,
            "created_at": self.created_at,
            "adapter": self.adapter,
            "attempt": self.attempt,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "transcript": self.transcript,
            "summary": self.summary(),
            "needs_recapture": self.needs_recapture,
            "recapture_reasons": list(self.recapture_reasons),
            "items": [r.to_dict() for r in self.results],
            "annotated_image_png_base64": (
                base64.b64encode(self.annotated_png).decode("ascii")
                if self.annotated_png
                else None
            ),
        }

    def to_html(self) -> str:
        return render_html(self)


def annotate_image(
    image: np.ndarray, results: list[ChecklistItemResult]
) -> bytes:
    """Draw verdict-coloured boxes for every evidence component."""
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for result in results:
        color = _STATUS_COLORS[result.status]
        for comp in result.evidence.components:
            box = comp.bbox
            if box is None:
                continue
            draw.rectangle(
                [box.x_min, box.y_min, box.x_max - 1, box.y_max - 1],
                outline=color, width=3,
            )
            draw.text(
                (box.x_min + 2, max(0, box.y_min - 12)),
                f"{comp.label} {result.status.value}",
                fill=color,
            )
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def build_report(
    *,
    spec_name: str,
    results: list[ChecklistItemResult],
    image: np.ndarray | None = None,
    transcript: str | None = None,
    adapter: str = "mock",
    attempt: int = 1,
    needs_recapture: bool = False,
    recapture_reasons: list[str] | None = None,
) -> Report:
    """Tool: build the evidence-linked report for one check run."""
    height = int(image.shape[0]) if image is not None else 0
    width = int(image.shape[1]) if image is not None else 0
    annotated = annotate_image(image, results) if image is not None else None
    return Report(
        spec_name=spec_name,
        results=results,
        image_width=width,
        image_height=height,
        annotated_png=annotated,
        transcript=transcript,
        adapter=adapter,
        attempt=attempt,
        needs_recapture=needs_recapture,
        recapture_reasons=list(recapture_reasons or []),
    )


def render_html(report: Report) -> str:
    rows: list[str] = []
    for r in report.results:
        measured = html.escape(str(r.measured))
        expected = html.escape(str(r.expected))
        tool = html.escape(r.evidence.tool)
        rows.append(
            "<tr>"
            f"<td>{html.escape(r.item_id)}</td>"
            f"<td>{html.escape(r.label)}</td>"
            f"<td>{html.escape(r.check)}</td>"
            f'<td class="status-{r.status.value}">{r.status.value}</td>'
            f"<td>{measured}</td><td>{expected}</td>"
            f"<td>{tool}</td>"
            f"<td>{html.escape(r.explanation)}</td>"
            "</tr>"
        )
    img_tag = ""
    if report.annotated_png:
        b64 = base64.b64encode(report.annotated_png).decode("ascii")
        img_tag = f'<img alt="Annotated capture" src="data:image/png;base64,{b64}">'
    transcript = (
        f"<p>Voice command: {html.escape(report.transcript)}</p>"
        if report.transcript
        else ""
    )
    recapture = ""
    if report.needs_recapture and report.recapture_reasons:
        items = "".join(f"<li>{html.escape(x)}</li>" for x in report.recapture_reasons)
        recapture = f"<h2>Please re-capture</h2><ul>{items}</ul>"
    summary = report.summary()
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>SpecCheck report - {html.escape(report.spec_name)}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem}"
        "table{border-collapse:collapse}td,th{border:1px solid #999;padding:.4rem .6rem}"
        ".status-PASS{color:#1a7a33;font-weight:700}.status-FAIL{color:#c02820;font-weight:700}"
        ".status-UNSURE{color:#b57a00;font-weight:700}img{max-width:100%}</style></head><body>"
        f"<h1>SpecCheck report: {html.escape(report.spec_name)}</h1>"
        f"<p>{report.created_at} | adapter: {html.escape(report.adapter)} | "
        f"attempt {report.attempt} | PASS {summary['PASS']} / FAIL {summary['FAIL']} / "
        f"UNSURE {summary['UNSURE']}</p>"
        f"{transcript}{recapture}{img_tag}"
        "<table><tr><th>Item</th><th>Label</th><th>Check</th><th>Status</th>"
        "<th>Measured</th><th>Expected</th><th>Tool</th><th>Explanation</th></tr>"
        + "".join(rows)
        + "</table></body></html>"
    )
