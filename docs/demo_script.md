# SpecCheck - 3-minute demo script

Setup (before the demo): `uv sync --extra dev`, then
`uv run uvicorn speccheck.api:app --port 8000`, open http://127.0.0.1:8000.
Say up front: "Real Meta adapters are wired behind these interfaces; today
the demo runs the mock adapters, and I will point at exactly where SAM 3.1,
Muse Spark, and Muse Voice Transcribe plug in."

## 0:00-0:30 - The problem
"When your hands are holding a PCB or halfway through an IKEA build, you
cannot type a checklist. SpecCheck lets you show the build to a camera, speak
a command, and get a verdict per checklist item - with evidence, not vibes."

## 0:30-1:15 - Happy path (PASS)
Click "Load sample: pass", keep the demo spec, click "Run check".
Point at: 4/4 PASS, the annotated image with boxes, and one row's measured
vs expected values plus mask evidence. "Every claim links to a mask, a
bounding box, and the tool that produced it. The counting is plain Python
on masks - no LLM guesses a number."

## 1:15-1:50 - Failure it catches (FAIL)
Click "Load sample: missing screw", run again. screws_count flips to FAIL
with measured 3 vs expected 4. "That is a real measurement disagreement, so
it is FAIL - not a shrug."

## 1:50-2:30 - Knowing when it does not know (UNSURE + recapture)
Click "Load sample: blank (UNSURE)", run. Everything is UNSURE and the
recapture banner explains why: nothing segmented. "SpecCheck never passes
an item without evidence. It asks you to re-capture, up to 3 attempts, and
says what to fix - angle, distance, occlusion."

## 2:30-3:00 - Voice, report, honesty
Click "Record voice command", stop, show the transcript field labelled as
mock. Download the HTML report. Close with limitations: "The mock segmenter
only reads synthetic colour-blob images; real photos need the SAM 3.1
adapter, which is a TODO stub until the verified Meta APIs land. Eval so
far is on 10 synthetic images - the numbers in the README are exactly what
eval/run_eval.py printed, nothing rounded up."
