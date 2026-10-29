"""FastAPI backend for SpecCheck.

Run: ``uv run uvicorn speccheck.api:app --port 8000`` then open
http://127.0.0.1:8000 in a browser.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

from . import agent as agent_module
from .adapters import build_adapters
from .config import get_settings
from .report import render_html
from .samples import DEMO_SPEC_YAML, SAMPLE_BUILDERS
from .spec import load_checklist_yaml
from .types import mask_from_rle
from .report import Report
from .types import ChecklistItemResult, CheckStatus, Evidence, SegmentedComponent

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

app = FastAPI(title="SpecCheck", version="0.1.0")


def _adapters():
    settings = get_settings()
    try:
        return build_adapters(settings.adapter), settings
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _decode_image(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as img:
            return np.asarray(img.convert("RGB"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"cannot decode image: {exc}") from exc


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "adapter": settings.adapter,
        "meta_configured": settings.meta_configured,
    }


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    settings = get_settings()
    return {
        "adapter": settings.adapter,
        "meta_configured": settings.meta_configured,
        "note": (
            "mock adapters are active; no Meta API is called"
            if settings.adapter == "mock"
            else "meta adapters selected but not wired yet (see DOCS_NOTES.md TODOs)"
        ),
    }


@app.get("/api/demo-spec")
def demo_spec() -> dict[str, str]:
    return {"yaml": DEMO_SPEC_YAML}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> dict[str, Any]:
    (_, _, transcriber), _ = _adapters()
    data = await audio.read()
    result = transcriber.transcribe(data, filename=audio.filename or "audio.webm")
    return {"text": result.text, "adapter": result.adapter, "mock": result.adapter.startswith("mock")}


@app.post("/api/check")
async def check(
    image: UploadFile = File(...),
    spec_yaml: str = Form(...),
    transcript: str = Form(""),
    attempt: int = Form(1),
) -> JSONResponse:
    (segmenter, reasoner, _), _ = _adapters()
    if attempt < 1 or attempt > agent_module.MAX_ATTEMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"attempt must be between 1 and {agent_module.MAX_ATTEMPTS}",
        )
    try:
        checklist = load_checklist_yaml(spec_yaml)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid spec: {exc}") from exc
    frame = _decode_image(await image.read())
    report = agent_module.run_check(
        frame,
        checklist,
        segmenter,
        reasoner,
        transcript=transcript or None,
        attempt=attempt,
    )
    return JSONResponse(report.to_dict())


@app.post("/api/report/html")
async def report_html(payload: dict[str, Any]) -> HTMLResponse:
    """Re-render a report JSON payload (from /api/check) as standalone HTML."""
    try:
        results = [_result_from_dict(item) for item in payload.get("items", [])]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid report payload: {exc}") from exc
    report = Report(
        spec_name=str(payload.get("spec_name", "report")),
        results=results,
        image_width=int(payload.get("image_width", 0)),
        image_height=int(payload.get("image_height", 0)),
        transcript=payload.get("transcript"),
        adapter=str(payload.get("adapter", "mock")),
        attempt=int(payload.get("attempt", 1)),
        needs_recapture=bool(payload.get("needs_recapture", False)),
        recapture_reasons=list(payload.get("recapture_reasons", [])),
    )
    return HTMLResponse(render_html(report))


def _result_from_dict(item: dict[str, Any]) -> ChecklistItemResult:
    components = []
    for comp in item.get("evidence", {}).get("components", []):
        rle = comp.get("mask_rle")
        if rle:
            components.append(
                SegmentedComponent(
                    label=comp["label"],
                    mask=mask_from_rle(rle),
                    confidence=float(comp.get("confidence", 0.0)),
                    prompt=comp.get("prompt", ""),
                )
            )
    return ChecklistItemResult(
        item_id=item["item_id"],
        label=item["label"],
        check=item["check"],
        status=CheckStatus(item["status"]),
        measured=item.get("measured", {}),
        expected=item.get("expected", {}),
        evidence=Evidence(
            tool=item.get("evidence", {}).get("tool", ""),
            components=components,
            notes=list(item.get("evidence", {}).get("notes", [])),
        ),
        explanation=item.get("explanation", ""),
        recapture_reason=item.get("recapture_reason"),
    )


@app.get("/api/sample/{name}")
def sample_image(name: str) -> FileResponse:
    builder = SAMPLE_BUILDERS.get(name)
    if builder is None:
        raise HTTPException(status_code=404, detail=f"unknown sample: {name}")
    out = Path(__file__).resolve().parent.parent / "samples" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(builder()).save(out)
    return FileResponse(out, media_type="image/png", filename=f"{name}.png")


@app.get("/")
def index() -> FileResponse:
    if not FRONTEND.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(FRONTEND, media_type="text/html")
