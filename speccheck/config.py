"""Runtime configuration from environment / .env.

Only ``META_API_KEY`` and ``SPECCHECK_ADAPTER`` are read. No key value is ever
logged or returned by the API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    meta_api_key: str | None
    adapter: str  # "mock" | "meta"

    @property
    def meta_configured(self) -> bool:
        return bool(self.meta_api_key)


def get_settings() -> Settings:
    _load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    key = os.environ.get("META_API_KEY") or None
    adapter = (os.environ.get("SPECCHECK_ADAPTER") or "mock").strip().lower()
    if adapter not in {"mock", "meta"}:
        adapter = "mock"
    return Settings(meta_api_key=key, adapter=adapter)
