# DOCS_NOTES - Meta Model API research (Step 0)

Status: **PAUSED by the user.** The verified APIs will be provided later.
Nothing below is filled in yet, and no endpoint, model ID, or parameter has
been invented anywhere in this codebase. The real adapters in
`speccheck/adapters/meta.py` raise `NotImplementedError` with TODOs until
each item here is confirmed from the official docs / cookbook.

## To confirm before wiring real adapters

- [ ] Exact model IDs for Muse Spark, SAM 3.1, Muse Voice Transcribe
      (source: https://dev.meta.ai/docs/models)
- [ ] Base URL, auth scheme, and how `META_API_KEY` is presented
      (source: https://dev.meta.ai/docs/overview, /docs/quickstart)
- [ ] How to send images (URL vs base64 vs multipart) and size/format limits
- [ ] Whether tool/function calling is supported, and its exact schema
- [ ] Whether structured JSON output is supported, and how to request it
- [ ] How to call SAM 3.1 with text prompts; mask format returned
      (polygon / RLE / image), confidence scores, multi-instance behaviour
- [ ] How to call Muse Voice Transcribe; accepted audio formats, language
      handling, streaming vs batch
- [ ] Rate limits, quotas, and error shapes for each model
- [ ] Cookbook patterns worth reusing
      (https://github.com/meta-models/meta-model-cookbook)

## Fallbacks if a capability is unclear or unsupported

- No tool/function calling -> the agent keeps its current design: the
  reasoner only explains deterministic verdicts and picks the next step from
  a fixed, code-enumerated set of tool names validated in Python.
- No structured JSON -> parse defensively, validate against the spec schema,
  and mark any unparseable interpretation UNSURE rather than guessing.
- No SAM 3.1 text prompts -> fall back to prompt-free segmentation plus a
  classifier/labelling step, or keep the mock segmenter for synthetic demos
  and say so in the README.
- No Voice Transcribe API -> browser Web Speech API or typed commands as the
  interim input path, clearly labelled as a substitute.
