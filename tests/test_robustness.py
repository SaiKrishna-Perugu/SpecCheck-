"""Mock segmenter must survive mild photo degradations of synthetic boards."""

import io

import numpy as np
from PIL import Image, ImageFilter

from speccheck import samples
from speccheck.adapters.mock import MockSegmenter
from speccheck.spec import load_checklist_yaml
from speccheck.tools import measure_count, segment

rng = np.random.default_rng(3)


def _seg(image):
    spec = load_checklist_yaml(samples.DEMO_SPEC_YAML)
    return segment(image, spec, MockSegmenter())


def test_gaussian_noise_keeps_count():
    noisy = np.clip(
        samples.sample_pass().astype(float) + rng.normal(0, 15, samples.sample_pass().shape),
        0, 255,
    ).astype(np.uint8)
    assert measure_count(_seg(noisy), "screw") == 4


def test_jpeg_artifacts_keep_count():
    buf = io.BytesIO()
    Image.fromarray(samples.sample_pass()).save(buf, format="JPEG", quality=35)
    buf.seek(0)
    degraded = np.asarray(Image.open(buf).convert("RGB"))
    assert measure_count(_seg(degraded), "screw") == 4


def test_darkened_image_keeps_count():
    dark = np.clip(samples.sample_pass().astype(float) * 0.55, 0, 255).astype(np.uint8)
    assert measure_count(_seg(dark), "screw") == 4


def test_blur_keeps_count():
    blurred = np.asarray(
        Image.fromarray(samples.sample_pass()).filter(ImageFilter.GaussianBlur(1.6))
    )
    assert measure_count(_seg(blurred), "screw") == 4


def test_unknown_colour_is_not_invented_as_part():
    image = samples.blank_image()
    image[40:70, 40:70] = (20, 200, 210)  # cyan blob: no palette prompt owns it
    seg = _seg(image)
    assert seg.components == []
