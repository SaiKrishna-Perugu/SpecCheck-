# SpecCheck

Voice-driven visual QA agent for hands-on assembly (PCBs, Lego/IKEA builds, wiring).
The user shows a photo or camera frame, speaks a command, and provides a
checklist/spec (YAML). SpecCheck verifies the assembly and returns an
evidence-linked report: every checklist item is PASS / FAIL / UNSURE with the
annotated image, mask evidence, measured values, and the tool that produced them.
If evidence is missing, the verdict is UNSURE - never PASS.

## What was built during the hackathon

Built fresh in this repo during the Meta Global AI Developer Hackathon:

- Typed tool layer (`segment`, `measure_count`, `measure_position`,
  `compare_to_spec`, `request_recapture`, `build_report`) in
  [speccheck/tools.py](speccheck/tools.py), each unit-tested.
- Deterministic Python checks on segmentation masks (count, relative position,
  area ratio, alignment, presence) in [speccheck/checks.py](speccheck/checks.py).
  No LLM decides anything code can compute.
- Agent loop in [speccheck/agent.py](speccheck/agent.py): transcribe -> segment ->
  deterministic checks -> plain-language explanation -> recapture request when
  confidence is low (max 3 attempts), then an evidence-linked HTML + JSON report.
- Meta model adapters behind clean interfaces
  ([speccheck/adapters/base.py](speccheck/adapters/base.py)): SAM 3.1
  (segmentation), Muse Spark (interpretation/explanation), Muse Voice
  Transcribe (speech). Mock implementations run today; the real adapters in
  [speccheck/adapters/meta.py](speccheck/adapters/meta.py) are explicit TODO
  stubs pending the verified API details in DOCS_NOTES.md - no endpoint or
  parameter has been invented.
- FastAPI backend ([speccheck/api.py](speccheck/api.py)) and a plain-HTML
  frontend ([frontend/index.html](frontend/index.html)) with webcam capture,
  image upload, and microphone recording.
- Tests in [tests/](tests/) covering the spec loader, checks, tools, agent
  loop (including recapture exhaustion), and the HTTP API.

## Run it

Requires Python 3.11+ (or [uv](https://docs.astral.sh/uv/)).

```powershell
uv sync --extra dev
uv run uvicorn speccheck.api:app --port 8000
```

Then open http://127.0.0.1:8000 in a browser. Try the built-in samples
("pass", "missing screw", "blank") with the demo spec, or paste your own YAML.

Run the tests:

```powershell
uv run pytest
```

## Voice commands

Typed or transcribed commands are parsed deterministically
([speccheck/intent.py](speccheck/intent.py)): "check the assembly" runs a
full check, "check screws count" runs only that item, "how many screws"
recounts, and "explain screws count" explains the last verdict with its
evidence. Unrecognised commands return `unknown` - the UI asks you to
rephrase instead of guessing. Re-captures advance the attempt counter
automatically (max 3). The image gallery in the UI lists every sample and
eval image. In production Muse Spark replaces this parser for ambiguous
phrasing; the deterministic parser stays as the offline fallback.

## Configuration

Copy `.env.example` to `.env`. `SPECCHECK_ADAPTER=mock` (default) uses the
offline mock models. `SPECCHECK_ADAPTER=fallback` uses a classical
colour-blob segmenter for real photos (spec items can carry a `color:`
hint); it is a SAM 3.1 substitute and labels itself as such, with modest
confidence. `SPECCHECK_ADAPTER=meta` uses the real Meta Model API
(`sam-3.1`, `muse-spark-1.3`, `muse-voice-transcribe-1.0`; see
DOCS_NOTES.md). Verified live: SAM decoded real masks (confidence ~0.93),
Spark explained check facts, Voice Transcribe answered a WAV upload.
Prompt wording matters: SAM matched "gray square" on the synthetic board
but not the bare word "screw", so checklist `prompt:` fields should name
concrete visible concepts. Never commit `.env` or a real `META_API_KEY`.

## Limitations (honest)

- The mock segmenter only finds flat-colour blobs in synthetic images. On real
  photos it returns nothing and every item is UNSURE. Real assembly photos
  need the SAM 3.1 adapter, which is not wired yet.
- The mock transcriber does not recognise speech; it returns a canned command
  and labels itself `mock` in the UI and API responses.
- The mock reasoner uses templates, not Muse Spark; ambiguous checklist
  wording is not yet reinterpreted by a model.
- Position checks compare centroids of the first target/reference instance;
  multi-instance spatial reasoning is out of scope.
- A fully occluded part is reported FAIL (missing), not UNSURE: the checks
  cannot distinguish "hidden" from "absent" (hard eval case h09). Partial
  occlusion works because some mask remains.
- Alignment is measured in image axes; rotating the whole board makes a
  well-aligned row read as misaligned (hard eval case h10).
- Eval (below) runs only on synthetic images drawn from the same colour
  palette the mock segmenter reads, so its scores are a pipeline sanity
  check, not real-world accuracy. Expect them to drop sharply on real
  photos until the SAM 3.1 adapter lands.

## Demo video

Animated walkthrough (3D orbit, scenarios, live check, classic UI):
[docs/speccheck-demo.mp4](docs/speccheck-demo.mp4) (56s, recorded from
the running app).

## Eval

10 synthetic images with labelled per-item outcomes live in
[eval/](eval/). Reproduce with:

```powershell
uv run python eval/generate_eval.py
uv run python eval/run_eval.py
```

Actual output of the last run (mock adapter, `eval/results.json`):
40/40 checklist items correct (item accuracy 1.0), 10/10 images fully
correct, UNSURE recall 1.0 (8/8 expected-UNSURE items), UNSURE precision
1.0 (8/8 predicted-UNSURE items), 0 false PASSes. Read the limitation
above before quoting these: the images are synthetic and palette-matched
to the mock segmenter by construction.

Hard set (`uv run python eval/run_eval.py --hard`, degraded/occluded
variants in `eval/hard/`): 47/48 items correct (accuracy 0.9792), 11/12
images fully correct, 0 false PASSes. The one miss is instructive: a
fully occluded chip is labelled UNSURE (it exists but is hidden) and the
agent says FAIL, because with the current checks a fully hidden part is
indistinguishable from a missing one. Noise, JPEG artifacts, dark/bright
exposure, blur, rotation, and half-occlusion all score correctly.

Photo-style set (`eval/realstyle/`, 30 labelled screenshots of the 3D
board renderer across scenarios and camera angles: perspective, shading,
shadows - harder than clip-art, still not phone photos), 120 items:

| Adapter | Accuracy | False PASS |
|---|---|---|
| mock (palette) | 0.1917 | 0 |
| fallback (classical) | 0.0667 | 0 |
| SAM 3.1 (`--adapter=meta`) | 0.3417 | 0 |

SAM is best but far from good on renders: it finds the chip reliably and
misses small screws and the LED/button relation. The fallback collapses
to UNSURE on rendered lighting. Per-adapter detail is in
`eval/realstyle/results-*.json`. Zero false PASSes at every level is the
property we keep; accuracy on real phone photos is the top remaining
work, and this set is the bridge measurement until those photos exist.

Hosting for judges (one-command local run, tunnel options, and why the
public demo must run mock mode): [docs/hosting.md](docs/hosting.md).

Baseline comparison on the standard set: Muse Spark alone judging the
photos (`scripts/baseline_vlm.py`) scored 0.75 accuracy with **2 false
PASSes** and UNSURE recall 0.625, versus the deterministic pipeline's
1.0 accuracy, 0 false PASSes, and UNSURE recall 1.0. That gap is the
argument for measuring with code and letting the model only explain.

## Architecture

Full diagram and component table: [docs/architecture.md](docs/architecture.md).
Demo walkthrough: [docs/demo_script.md](docs/demo_script.md).
M1 smoke script (one call per adapter): `uv run python scripts/meta_smoke.py`.

```
voice -> [Transcriber adapter] -> transcript/intent
photo -> [Segmenter adapter: SAM 3.1] -> masks + confidence
masks + YAML spec -> deterministic checks (count/position/area/alignment)
verdicts + evidence -> [Reasoner adapter: Muse Spark] plain-language summary
low confidence -> request_recapture (max 3 attempts)
all verdicts -> evidence-linked report (HTML + JSON)
```
