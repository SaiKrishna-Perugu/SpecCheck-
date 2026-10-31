# DOCS_NOTES - Meta Model API research (Step 0, completed)

Researched from the official docs (dev.meta.ai) after the user supplied
`META_API_KEY` in `.env`. Every claim below was read on the cited page;
anything not confirmed is marked as such. No endpoint or parameter in the
codebase comes from anywhere else.

Sources: /docs/overview, /docs/quickstart, /docs/models,
/docs/sam/segmenting, /docs/sam/reading-segmentation,
/docs/sam/client-libraries, /docs/speech-to-text,
/docs/api-reference/voice/transcribe, /docs/api-reference/voice/schemas,
/docs/pricing-rate-limits, and the cookbook repo
github.com/meta-models/meta-model-cookbook (agentic workflow recipes).

## Base URL and authentication

- Base URL: `https://api.meta.ai/v1`
- Auth: `Authorization: Bearer <key>` (docs examples name the env var
  `MODEL_API_KEY`; SpecCheck's `.env` contract uses `META_API_KEY` per the
  original brief, and the adapters read that name first)
- `GET /v1/models` lists the hosted catalog; unknown model returns `404`

## Exact model IDs (/docs/models)

| Role in SpecCheck | Model ID | Notes |
|---|---|---|
| Reasoning / explanation | `muse-spark-1.3` | Recommended for new work; 1M context; text/image/video/audio/PDF in, text out |
| Segmentation | `sam-3.1` | Text prompt + image/video in; boxes and masks out |
| Speech to text | `muse-voice-transcribe-1.0` | Audio in; transcript text out |

Also hosted: `muse-wonder-1.0`, `muse-image-1.0`. Contributor variants
(e.g. `muse-spark-1.3-contributor`) trade data-use permission for price.

## How to send images

Responses API (`POST /v1/responses`): one user message whose content holds
an `input_text` part and an `input_image` part. `image_url` accepts a
public URL or a base64 data URL (`data:image/png;base64,...`). Chat
Completions and Messages protocols exist on the same base URL; SpecCheck
uses Responses throughout. Muse Spark takes the same shape for image
understanding.

## Tool/function calling and structured JSON output

Both are supported capabilities of Muse Spark / Muse Wonder (dedicated
/docs/tool-calling and /docs/structured-output guides; the models page
lists both per model). SpecCheck does not depend on them: verdicts are
computed deterministically, the intent parser is rule-based with the
model as explainer only. If Muse Spark tool calling is adopted later,
the existing typed tools map 1:1 onto function definitions. Limitation:
the exact tool-schema JSON was not re-derived here because the design
does not need it yet.

## How to call SAM 3.1 (/docs/sam/segmenting, /docs/sam/reading-segmentation)

- `POST /v1/responses` with `"model": "sam-3.1"`, one user message:
  `input_text` = ONE short noun phrase concept (`"red square"`), plus
  `input_image`. One concept per request; lists return zero matches.
  No negation/spatial phrasing; filter by box coordinates in app code.
- `"stream": true` streams one line per frame; non-streaming returns the
  same format in one response. Zero matches = completed stream with
  empty `output_text` (normal, not an error).
- `"metadata": {"include_confidence": "true"}` (string, not boolean)
  adds `c` (0..1) to every box/mask. `"metadata": {"mask_encoding":
  "one_bit"}` selects compact `!` masks; default lossless is `~`.
- Output text grammar per frame:
  `<0f>id<|box;x1=..;y1=..;x2=..;y2=..;w=W;h=H[;c=..]|><|mask;x=0;y=0[;c=..];data=H,W,<enc><base85 payload>|>,...`
  Boxes are source pixels, x2/y2 inclusive. Mask raster is H rows x W
  cols (its own size, smaller than the frame), payload is base85.
- Decode with the official Python parser: `pip install meta-sam-parser`
  (>= 0.0.6 when confidence is requested): `decode_mask_to_raster`,
  `decode_mask_to_rle`, `decode_mask_to_svg_path`. SpecCheck pins this
  as the optional `meta` extra and places decoded rasters at their box.
- Errors: malformed prompts/metadata return permanent `400`s; do not
  retry unchanged. Image requests return every matching instance.

## How to call Muse Voice Transcribe (/docs/api-reference/voice/transcribe)

- File path: `POST /v1/asr/transcribe` as `multipart/form-data` with a
  `request` part (JSON: `{"model": "muse-voice-transcribe-1.0",
  "audioEncoding": "WAV", "mode": "PUSH_TO_TALK"}`) and an `audio` part.
- WAV only: non-WAV returns `400`; max 10 minutes / 32 MB. Response JSON
  (Accept: application/json) carries `transcript` (whole clip),
  `audioDurationMs`, `turns` (empty in PUSH_TO_TALK), and `sessionId`.
- Realtime alternative: `wss://api.meta.ai/v1/asr/realtime` (not used;
  the app records-then-uploads). Consequence: browser WebM recordings
  must be converted to WAV (the adapter converts via ffmpeg when present
  and otherwise fails with that exact message) or the user types instead.
- 25 evaluated languages, diarization/endpointing in other modes; no
  word-level timestamps.

## Pricing and rate limits (/docs/pricing-rate-limits, per team)

| Model | Price | Limits |
|---|---|---|
| `muse-spark-1.3` | $1.25 in / $4.25 out per 1M tokens (Standard) | 3,000 RPM, 4M TPM |
| `sam-3.1` | $2.50 per 1,000 images | 3,000 RPM, no TPM |
| `muse-voice-transcribe-1.0` | $0.18 per audio hour | 256 concurrent streams, 16,000 streams/hour |

On `429`: exponential backoff with jitter, honor `Retry-After`.
Background responses capped at 600 submissions/min/team.

## Live verification (first real calls)

- `muse-voice-transcribe-1.0` on 1s of WAV silence returned an empty
  transcript with HTTP 200 (endpoint, auth, multipart shape confirmed).
- `muse-spark-1.3` returned a plain-language explanation of the check
  facts via `/v1/responses`.
- `sam-3.1` prompt behaviour on the synthetic demo board: `"screw"`
  returned zero matches (empty `output_text`, status completed), while
  `"gray square"` returned all 4 squares (boxes exact, mask area 324px,
  confidence ~0.93, decoded with `meta-sam-parser`). Lesson, matching
  the docs: prompts must name concrete visible concepts; checklist
  `prompt:` fields should describe the part as it looks ("metal screw",
  "red LED"), especially for real photos vs clip-art.

## Fallbacks (kept, by design)

- Adapter modes `mock` and `fallback` remain fully offline paths; `meta`
  failures surface as explicit API errors, never silent mock substitution.
- If `meta-sam-parser` is unavailable, SAM output cannot be decoded
  safely, so the adapter raises instead of guessing a mask format.
