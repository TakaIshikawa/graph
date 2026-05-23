"""Small helpers shared by focused CSV report exporters."""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

WHITESPACE_RE = re.compile(r"\s+")


def render_csv(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_csv(path: str | Path, text: str) -> tuple[str, int]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return str(output_path), output_path.stat().st_size


def get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def metadata(value: object) -> Mapping[str, Any]:
    raw = get(value, "metadata")
    return raw if isinstance(raw, Mapping) else {}


def field_value(value: object) -> str:
    return inline_text(getattr(value, "value", value))


def inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return WHITESPACE_RE.sub(" ", text).strip()


def normalized_key(value: object) -> str:
    return field_value(value).casefold().replace("-", "_").replace(" ", "_")


def sort_key(value: object) -> tuple[str, str]:
    text = inline_text(value)
    return (text.casefold(), text)


def flatten_values(value: object) -> list[object]:
    if value is None or isinstance(value, bytes):
        return []
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in flatten_values(child)]
    if isinstance(value, list | tuple | set):
        return [item for child in value for item in flatten_values(child)]
    return [value]


def object_id(value: object, *keys: str) -> str:
    for key in keys:
        text = field_value(get(value, key))
        if text:
            return text
    return ""


def unit_id(unit: object) -> str:
    return object_id(unit, "id", "unit_id", "source_id")


def source_id(source: object) -> str:
    return object_id(source, "id", "source_id", "key")


def edge_id(edge: object) -> str:
    return object_id(edge, "id", "edge_id", "relation_id")


def parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = field_value(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
