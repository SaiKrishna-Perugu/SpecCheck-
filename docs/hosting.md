# Hosting SpecCheck for judges

## One-command local run (no key needed)

```powershell
uv sync --extra dev
$env:SPECCHECK_ADAPTER = "mock"
uv run uvicorn speccheck.api:app --port 8000
```

Open http://127.0.0.1:8000 (classic UI) or http://127.0.0.1:8000/3d
(3D board view). Mock mode needs no API key and works offline on the
sample and eval images.

## Public demo (pick one)

Neither tunnel tool is installed on the dev machine, so no public URL is
claimed yet. Either command creates one in under a minute:

```powershell
# Option A: Cloudflare quick tunnel (no account)
cloudflared tunnel --url http://127.0.0.1:8000

# Option B: nZucchini
nZucchini http 8000
```

Host the **mock** adapter publicly. Never expose a server running
`SPECCHECK_ADAPTER=meta` with your key to the open internet; run meta
mode only on your own machine or behind access control.

## Live pitch configuration

For the in-person demo, run meta mode locally so SAM 3.1, Muse Spark,
and Voice Transcribe answer for real:

```powershell
$env:SPECCHECK_ADAPTER = "meta"   # key comes from .env, never committed
uv run uvicorn speccheck.api:app --port 8000
```
