"""Voice-command intent parsing (deterministic, no LLM).

Turns a transcript into one structured action the agent can execute:
full check, check one item, recount one label, or explain a result.
In production Muse Spark handles ambiguous phrasing; this rule-based
parser is the offline fallback and never guesses: unknown input maps to
``unknown`` and the UI asks the user to rephrase or tap Run check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .spec import Checklist


@dataclass(frozen=True)
class Intent:
    action: str  # full_check | check_item | recount | explain | unknown
    item_id: str | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "item_id": self.item_id, "label": self.label}


_EXPLAIN_WORDS = ("explain", "why", "what happened", "tell me about")
_RECOUNT_WORDS = ("recount", "count again", "how many")
_CHECK_WORDS = ("check", "verify", "inspect", "look at", "run")


def _match_item(text: str, checklist: Checklist) -> tuple[str | None, str | None]:
    """Best (item_id, label) whose id/label words appear in the text."""
    best: tuple[int, str, str] | None = None
    for item in checklist.items:
        candidates = {item.id.replace("_", " "), item.label}
        for phrase in candidates:
            phrase = phrase.lower()
            if phrase and phrase in text:
                score = len(phrase)
                if best is None or score > best[0]:
                    best = (score, item.id, item.label)
    if best is None:
        return None, None
    return best[1], best[2]


def parse_intent(transcript: str | None, checklist: Checklist) -> Intent:
    text = re.sub(r"\s+", " ", (transcript or "").strip().lower())
    if not text:
        return Intent(action="unknown")
    item_id, label = _match_item(text, checklist)
    if any(w in text for w in _EXPLAIN_WORDS) and item_id:
        return Intent(action="explain", item_id=item_id, label=label)
    if any(w in text for w in _RECOUNT_WORDS):
        return Intent(action="recount", item_id=item_id, label=label or _singular(text))
    if item_id and any(w in text for w in _CHECK_WORDS):
        return Intent(action="check_item", item_id=item_id, label=label)
    if any(w in text for w in ("check", "verify", "run", "inspect", "assembly", "spec")):
        return Intent(action="full_check")
    return Intent(action="unknown", item_id=item_id, label=label)


def _singular(text: str) -> str | None:
    match = re.search(r"how many ([a-z]+)", text)
    if not match:
        return None
    word = match.group(1)
    return word[:-1] if word.endswith("s") else word
