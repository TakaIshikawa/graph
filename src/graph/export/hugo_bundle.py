"""Hugo Markdown content bundle export helpers."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from graph.types.models import KnowledgeUnit

_URL_KEYS = ("external_url", "source_url", "url", "uri")


def export_units_to_hugo_bundle(
    units: Iterable[KnowledgeUnit],
    output_dir: str | Path,
    *,
    include_index: bool = False,
) -> dict[str, Any]:
    """Write one Hugo-compatible Markdown file per unit."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_units = list(units)
    exported_units = all_units if isinstance(units, Sequence) else sorted(all_units, key=_unit_sort_key)
    slug_counts = Counter(_slug(unit.title) for unit in exported_units)
    files: list[Path] = []

    for unit in exported_units:
        slug = _slug(unit.title)
        if slug_counts[slug] > 1:
            slug = f"{slug}-{_slug(unit.source_id) or _stable_suffix(unit)}"
        path = output_path / f"{slug}.md"
        path.write_text(_markdown_file(unit), encoding="utf-8")
        files.append(path)

    index_path: Path | None = None
    if include_index:
        index_path = output_path / "_index.md"
        index_path.write_text(_index_markdown(exported_units), encoding="utf-8")

    return {
        "path": str(output_path),
        "units_scanned": len(all_units),
        "units_exported": len(exported_units),
        "files_written": len(files) + (1 if index_path else 0),
        "index_written": index_path is not None,
    }


def _markdown_file(unit: KnowledgeUnit) -> str:
    frontmatter: dict[str, Any] = {
        "title": unit.title,
        "date": _yaml_value(unit.created_at),
        "lastmod": _yaml_value(unit.updated_at),
        "tags": sorted(_clean_text(tag) for tag in unit.tags if _clean_text(tag)),
        "source_project": _yaml_value(unit.source_project),
        "source_id": unit.source_id,
        "metadata": _yaml_value(unit.metadata),
    }
    if external_url := _first_text(unit.metadata, _URL_KEYS):
        frontmatter["external_url"] = external_url
    yaml_text = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{yaml_text}\n---\n\n{unit.content or ''}\n"


def _index_markdown(units: Sequence[KnowledgeUnit]) -> str:
    grouped: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        grouped[str(_yaml_value(unit.source_project) or "")].append(unit)

    lines = ["---", "title: Knowledge Index", "---", ""]
    for project in sorted(grouped):
        lines.extend([f"## {_markdown_text(project or 'unknown')}", ""])
        for unit in sorted(grouped[project], key=_unit_sort_key):
            lines.append(f"- {_markdown_text(unit.title)} ({_markdown_text(unit.source_id)})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _slug(value: object) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "untitled"


def _stable_suffix(unit: KnowledgeUnit) -> str:
    seed = f"{_yaml_value(unit.source_project)}:{unit.source_id}:{unit.id}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]


def _first_text(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in metadata and (text := _clean_text(metadata.get(key))):
            return text
    return ""


def _yaml_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return _yaml_value(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): _yaml_value(item) for key, item in sorted(value.items(), key=_item_key)}
    if isinstance(value, list | tuple | set):
        return [_yaml_value(item) for item in value]
    return str(value)


def _markdown_text(value: object) -> str:
    return _clean_text(value).replace("[", "\\[").replace("]", "\\]")


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r\n", "\n").replace("\r", "\n").split())


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (str(_yaml_value(unit.source_project) or ""), str(unit.source_id or ""), str(unit.title or ""))


def _item_key(item: tuple[Any, Any]) -> str:
    return str(item[0])
