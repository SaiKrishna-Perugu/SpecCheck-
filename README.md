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

## Configuration

Copy `.env.example` to `.env`. `SPECCHECK_ADAPTER=mock` (default) uses the
offline mock models. `SPECCHECK_ADAPTER=meta` selects the real Meta adapters,
which currently raise `NotImplementedError` with TODOs until DOCS_NOTES.md is
completed from the official docs. Never commit `.env` or a real
`META_API_KEY`.

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
- Eval (below) runs only on synthetic images drawn from the same colour
  palette the mock segmenter reads, so its scores are a pipeline sanity
  check, not real-world accuracy. Expect them to drop sharply on real
  photos until the SAM 3.1 adapter lands.

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
