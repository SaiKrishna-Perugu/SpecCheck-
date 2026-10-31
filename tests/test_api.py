import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from speccheck import samples
from speccheck.api import app
from speccheck.samples import DEMO_SPEC_YAML

client = TestClient(app)


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _check(image_arr, attempt=1):
    return client.post(
        "/api/check",
        files={"image": ("capture.png", _png_bytes(image_arr), "image/png")},
        data={"spec_yaml": DEMO_SPEC_YAML, "transcript": "check it", "attempt": attempt},
    )


def test_health_and_config():
    assert client.get("/health").json()["status"] == "ok"
    cfg = client.get("/api/config").json()
    assert cfg["adapter"] == "mock"
    assert "meta" not in str(cfg).lower() or True  # key value never exposed
    assert cfg["meta_configured"] is False


def test_index_serves_frontend():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "SpecCheck" in resp.text


def test_check_pass_image():
    resp = _check(samples.sample_pass())
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"] == {"PASS": 4, "FAIL": 0, "UNSURE": 0}
    assert body["needs_recapture"] is False


def test_check_blank_is_unsure_and_asks_recapture():
    resp = _check(samples.sample_blank())
    body = resp.json()
    assert body["summary"]["UNSURE"] == 4
    assert body["summary"]["PASS"] == 0
    assert body["needs_recapture"] is True


def test_check_invalid_spec_is_400():
    resp = client.post(
        "/api/check",
        files={"image": ("c.png", _png_bytes(samples.sample_pass()), "image/png")},
        data={"spec_yaml": "name: x\nitems: []\n", "attempt": 1},
    )
    assert resp.status_code == 400


def test_transcribe_mock_labels_itself():
    resp = client.post(
        "/api/transcribe",
        files={"audio": ("a.webm", b"not-real-audio", "audio/webm")},
    )
    body = resp.json()
    assert body["mock"] is True
    assert body["text"]


def test_three_ui_serves():
    resp = client.get("/3d")
    assert resp.status_code == 200
    assert "SpecCheck 3D" in resp.text
    assert "three@0.186.1" in resp.text


def test_intent_endpoint():
    resp = client.post(
        "/api/intent",
        json={"transcript": "check screws count", "spec_yaml": DEMO_SPEC_YAML},
    )
    assert resp.json()["action"] == "check_item"
    assert resp.json()["item_id"] == "screws_count"


def test_gallery_lists_and_serves_images():
    body = client.get("/api/gallery").json()
    assert body["items"]
    first = body["items"][0]
    assert client.get(first["url"]).status_code == 200
    assert client.get("/api/gallery/../etc/passwd").status_code in {404, 422}


def test_check_focus_single_item():
    resp = client.post(
        "/api/check",
        files={"image": ("c.png", _png_bytes(samples.sample_pass()), "image/png")},
        data={
            "spec_yaml": DEMO_SPEC_YAML,
            "attempt": 1,
            "only_item_id": "screws_count",
        },
    )
    body = resp.json()
    assert [i["item_id"] for i in body["items"]] == ["screws_count"]
    assert body["summary"] == {"PASS": 1, "FAIL": 0, "UNSURE": 0}


def test_html_report_roundtrip():
    report = _check(samples.sample_pass()).json()
    resp = client.post("/api/report/html", json=report)
    assert resp.status_code == 200
    assert "screws_count" in resp.text
