"""Small helpers for personal data export adapters."""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def iter_paths(path: str, suffixes: set[str]) -> list[Path]:
    if not path:
        return []
    root = Path(path).expanduser()
    suffixes = {suffix.lower() for suffix in suffixes}
    if root.is_file() and root.suffix.lower() in suffixes:
        return [root]
    if not root.is_dir():
        return []
    return sorted(child for child in root.rglob("*") if child.is_file() and child.suffix.lower() in suffixes)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        return [{str(key).strip(): value for key, value in row.items() if key is not None} for row in reader]


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def first(row: dict[str, Any], *keys: str) -> str:
    compact = {normalize_key(str(key)): value for key, value in row.items()}
    for key in keys:
        value = row.get(key)
        if value is None:
            value = compact.get(normalize_key(key))
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def split_values(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = re.split(r"[,;|]", "" if value is None else str(value))
    items: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in items:
            items.append(text)
    return items


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_datetime(value: Any) -> datetime | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00"), f"{text}T00:00:00"):
        try:
            return ensure_utc(datetime.fromisoformat(candidate))
        except ValueError:
            pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
        "%b %d, %Y",
        "%B %d, %Y",
    ):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def parse_float(value: Any) -> float | None:
    text = "" if value is None else str(value).strip().replace(",", "")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    number = parse_float(value)
    return int(number) if number is not None else None


def parse_money(value: Any) -> float | None:
    return parse_float(value)


def parse_duration_seconds(value: Any) -> int | None:
    text = "" if value is None else str(value).strip().lower()
    if not text:
        return None
    if re.fullmatch(r"\d+(?::\d{1,2}){1,2}", text):
        parts = [int(part) for part in text.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    number = parse_float(text)
    if number is None:
        return None
    if "hour" in text or re.search(r"\bhrs?\b", text):
        return int(round(number * 3600))
    if "min" in text:
        return int(round(number * 60))
    return int(round(number))


def digest_source_id(prefix: str, *parts: Any) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if value not in ("", None, [])}
