"""Checklist / spec loading from YAML.

Example::

    name: demo-board
    items:
      - id: screws_count
        label: screw
        check: count
        expected_count: 4
      - id: led_left_of_button
        label: led
        check: position
        relation: left_of
        reference_label: button
      - id: chip_area
        label: chip
        check: area_ratio
        reference_label: board
        min_ratio: 0.02
        max_ratio: 0.30
      - id: screws_aligned
        label: screw
        check: alignment
        axis: y
        max_spread_px: 12
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

CHECK_TYPES = {"count", "presence", "position", "area_ratio", "alignment"}
RELATIONS = {"left_of", "right_of", "above", "below"}
AXES = {"x", "y", "line"}


@dataclass(frozen=True)
class ChecklistItem:
    id: str
    label: str
    check: str
    prompt: str = ""  # text prompt for the segmenter; defaults to label
    color: str | None = None  # optional colour hint, e.g. "red" (fallback segmenter)
    expected_count: int | None = None
    relation: str | None = None
    reference_label: str | None = None
    reference_prompt: str = ""
    min_ratio: float | None = None
    max_ratio: float | None = None
    axis: str | None = None
    max_spread_px: float | None = None
    min_confidence: float = 0.5
    min_area_fraction: float = 0.0005

    def __post_init__(self) -> None:
        if not self.prompt:
            object.__setattr__(self, "prompt", self.label)
        if self.reference_label and not self.reference_prompt:
            object.__setattr__(self, "reference_prompt", self.reference_label)


@dataclass(frozen=True)
class Checklist:
    name: str
    items: tuple[ChecklistItem, ...] = field(default_factory=tuple)

    def prompts(self) -> list[str]:
        seen: list[str] = []
        for item in self.items:
            for p in (item.prompt, item.reference_prompt):
                if p and p not in seen:
                    seen.append(p)
        return seen

    def color_hints(self) -> dict[str, str]:
        hints: dict[str, str] = {}
        for item in self.items:
            if item.color:
                hints.setdefault(item.prompt, item.color)
        return hints

    def prompt_to_label(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for item in self.items:
            mapping.setdefault(item.prompt, item.label)
            if item.reference_label:
                mapping.setdefault(item.reference_prompt, item.reference_label)
        return mapping


def _parse_item(raw: dict[str, Any], index: int) -> ChecklistItem:
    if not isinstance(raw, dict):
        raise ValueError(f"items[{index}] must be a mapping")
    item_id = str(raw.get("id") or f"item_{index + 1}")
    label = raw.get("label")
    check = raw.get("check")
    if not label or not isinstance(label, str):
        raise ValueError(f"items[{index}] ({item_id}): 'label' is required")
    if check not in CHECK_TYPES:
        raise ValueError(
            f"items[{index}] ({item_id}): 'check' must be one of {sorted(CHECK_TYPES)}, got {check!r}"
        )
    relation = raw.get("relation")
    if check == "position":
        if relation not in RELATIONS:
            raise ValueError(
                f"item {item_id!r}: position check requires 'relation' in {sorted(RELATIONS)}"
            )
        if not raw.get("reference_label"):
            raise ValueError(f"item {item_id!r}: position check requires 'reference_label'")
    axis = raw.get("axis")
    if check == "alignment" and axis not in AXES:
        raise ValueError(f"item {item_id!r}: alignment check requires 'axis' of 'x' or 'y'")
    if check == "count" and raw.get("expected_count") is None:
        raise ValueError(f"item {item_id!r}: count check requires 'expected_count'")
    if check == "area_ratio" and raw.get("min_ratio") is None and raw.get("max_ratio") is None:
        raise ValueError(f"item {item_id!r}: area_ratio check requires 'min_ratio' and/or 'max_ratio'")
    return ChecklistItem(
        id=item_id,
        label=str(label),
        check=str(check),
        prompt=str(raw.get("prompt") or label),
        color=str(raw["color"]).lower() if raw.get("color") else None,
        expected_count=(
            int(raw["expected_count"]) if raw.get("expected_count") is not None else None
        ),
        relation=str(relation) if relation else None,
        reference_label=str(raw["reference_label"]) if raw.get("reference_label") else None,
        reference_prompt=str(raw.get("reference_prompt") or raw.get("reference_label") or ""),
        min_ratio=float(raw["min_ratio"]) if raw.get("min_ratio") is not None else None,
        max_ratio=float(raw["max_ratio"]) if raw.get("max_ratio") is not None else None,
        axis=str(axis) if axis else None,
        max_spread_px=(
            float(raw["max_spread_px"]) if raw.get("max_spread_px") is not None else None
        ),
        min_confidence=float(raw.get("min_confidence", 0.5)),
        min_area_fraction=float(raw.get("min_area_fraction", 0.0005)),
    )


def parse_checklist(data: dict[str, Any]) -> Checklist:
    if not isinstance(data, dict):
        raise ValueError("spec must be a YAML mapping with 'name' and 'items'")
    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("spec requires a non-empty 'items' list")
    items = tuple(_parse_item(raw, i) for i, raw in enumerate(raw_items))
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("checklist item ids must be unique")
    return Checklist(name=str(data.get("name") or "unnamed-spec"), items=items)


def load_checklist_yaml(text: str) -> Checklist:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
    if data is None:
        raise ValueError("spec YAML is empty")
    return parse_checklist(data)
