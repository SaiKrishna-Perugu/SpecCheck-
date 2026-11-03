"""End-to-end voice verification with synthesized speech (real audio path).

Generates one WAV per intent with Windows SAPI text-to-speech, uploads each
to /api/transcribe on a running meta-mode server, parses intent via
/api/intent, and checks the action matches. This exercises the real Muse
Voice Transcribe API with real speech audio (synthesized, not recorded).

Prereq: a server in meta mode, e.g. SPECCHECK_ADAPTER=meta on port 8001.
Run: ``uv run python scripts/verify_voice_e2e.py [base_url]``
WAVs are written to .tmp/voice/ (gitignored scratch).
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"

COMMANDS = [
    ("check the assembly against the spec", "full_check"),
    ("check screws count", "check_item"),
    ("how many screws are there", "recount"),
    ("explain screws count", "explain"),
]

PS_TEMPLATE = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SetOutputToWaveFile('{path}')
$s.Speak('{text}')
$s.Dispose()
"""


def make_wav(text: str, path: Path) -> None:
    script = PS_TEMPLATE.format(path=str(path).replace("'", "''"), text=text.replace("'", "''"))
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0 or not path.exists():
        raise RuntimeError(f"TTS failed for {text!r}: {proc.stderr[:300]}")
    # The API expects mono s16le at 16000 or 24000 Hz; SAPI emits 22050 Hz.
    import audioop

    with wave.open(str(path), "rb") as wav:
        rate, width, frames = wav.getframerate(), wav.getsampwidth(), wav.readframes(wav.getnframes())
    if rate != 16000:
        frames, _ = audioop.ratecv(frames, width, 1, rate, 16000, None)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(width)
            wav.setframerate(16000)
            wav.writeframes(frames)


def post_multipart(url: str, field: str, filename: str, data: bytes) -> dict:
    boundary = "voice-e2e-boundary"
    body = (
        f"--{boundary}\r\n".encode()
        + f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode()
        + b"Content-Type: audio/wav\r\n\r\n" + data + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    from speccheck.samples import DEMO_SPEC_YAML

    out_dir = ROOT / ".tmp" / "voice"
    out_dir.mkdir(parents=True, exist_ok=True)
    passed = 0
    for i, (text, expected) in enumerate(COMMANDS):
        wav_path = out_dir / f"cmd{i}.wav"
        make_wav(text, wav_path)
        heard = post_multipart(f"{BASE}/api/transcribe", "audio", wav_path.name, wav_path.read_bytes())
        intent = post_json(f"{BASE}/api/intent", {"transcript": heard.get("text", ""), "spec_yaml": DEMO_SPEC_YAML})
        ok = intent.get("action") == expected
        passed += int(ok)
        print(f"{'ok ' if ok else 'MISS'} said={text!r} heard={heard.get('text')!r} "
              f"intent={intent.get('action')} expected={expected}")
    print(f"voice e2e: {passed}/{len(COMMANDS)} intents correct via {BASE}")
    return 0 if passed == len(COMMANDS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
