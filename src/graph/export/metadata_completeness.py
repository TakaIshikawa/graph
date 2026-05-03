"""Markdown metadata completeness reports for knowledge units."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


def export_metadata_completeness_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path,
    *,
    required_keys: Sequence[str] | None = None,
    optional_keys: Sequence[str] | None = None,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Write a deterministic Markdown report of unit metadata key completeness."""
    if (
        not isinstance(max_examples, int)
        or isinstance(max_examples, bool)
        or max_examples < 0
    ):
        raise ValueError("max_examples must be a non-negative integer")

    all_units = sorted(list(units), key=_unit_sort_key)
    required = _normalized_keys(required_keys)
    optional = _normalized_keys(optional_keys)
    if optional_keys is None:
        observed = {
            _inline_text(key)
            for unit in all_units
            for key in unit.metadata
            if _inline_text(key)
        }
        optional = sorted(observed.difference(required), key=_text_sort_key)

    stats = _build_stats(all_units, required, optional, Path(path), max_examples)
    report = _render_report(stats)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    return {
        "path": str(output_path),
        "units_scanned": stats["units_scanned"],
        "required_keys": [*required],
        "optional_keys": [*optional],
        "missing_required_counts": dict(stats["missing_required_counts"]),
    }


def _build_stats(
    units: list[KnowledgeUnit],
    required_keys: list[str],
    optional_keys: list[str],
    path: Path,
    max_examples: int,
) -> dict[str, Any]:
    required_present = _presence_counts(units, required_keys)
    optional_present = _presence_counts(units, optional_keys)
    missing_required_counts = {
        key: len(units) - required_present[key] for key in required_keys
    }
    missing_examples = {
        key: [
            _unit_label(unit)
            for unit in sorted(
                (unit for unit in units if not _has_metadata_value(unit.metadata, key)),
                key=_unit_sort_key,
            )[:max_examples]
        ]
        for key in required_keys
        if missing_required_counts[key]
    }
    units_missing_required = sum(
        1
        for unit in units
        if any(not _has_metadata_value(unit.metadata, key) for key in required_keys)
    )

    return {
        "path": str(path),
        "units_scanned": len(units),
        "required_keys": required_keys,
        "optional_keys": optional_keys,
        "required_present_counts": required_present,
        "optional_present_counts": optional_present,
        "missing_required_counts": missing_required_counts,
        "missing_required_examples": missing_examples,
        "units_missing_required": units_missing_required,
    }


def _render_report(stats: Mapping[str, Any]) -> str:
    units_scanned = stats["units_scanned"]
    lines = [
        "# Unit Metadata Completeness",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Units scanned | {units_scanned} |",
        f"| Required keys | {len(stats['required_keys'])} |",
        f"| Optional keys | {len(stats['optional_keys'])} |",
        f"| Units missing required keys | {stats['units_missing_required']} |",
        "",
        "## Required Key Coverage",
        "",
        "| Key | Present | Missing | Coverage |",
        "| --- | ---: | ---: | ---: |",
    ]

    if stats["required_keys"]:
        for key in stats["required_keys"]:
            present = stats["required_present_counts"][key]
            missing = stats["missing_required_counts"][key]
            lines.append(
                "| "
                f"{_markdown_cell(key)} | "
                f"{present} | "
                f"{missing} | "
                f"{_coverage_percent(present, units_scanned)} |"
            )
    else:
        lines.append("| _None_ | 0 | 0 | 0.0% |")

    lines.extend(
        [
            "",
            "## Optional Key Coverage",
            "",
            "| Key | Present | Missing | Coverage |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    if stats["optional_keys"]:
        for key in stats["optional_keys"]:
            present = stats["optional_present_counts"][key]
            lines.append(
                "| "
                f"{_markdown_cell(key)} | "
                f"{present} | "
                f"{units_scanned - present} | "
                f"{_coverage_percent(present, units_scanned)} |"
            )
    else:
        lines.append("| _None_ | 0 | 0 | 0.0% |")

    lines.extend(
        [
            "",
            "## Missing Required Key Examples",
            "",
            "| Key | Missing units | Examples |",
            "| --- | ---: | --- |",
        ]
    )
    missing_counts = stats["missing_required_counts"]
    examples = stats["missing_required_examples"]
    missing_keys = [key for key in stats["required_keys"] if missing_counts[key]]
    if missing_keys:
        for key in missing_keys:
            lines.append(
                "| "
                f"{_markdown_cell(key)} | "
                f"{missing_counts[key]} | "
                f"{_markdown_cell('; '.join(examples.get(key, [])))} |"
            )
    else:
        lines.append("| _None_ | 0 | _None_ |")

    return "\n".join(lines).rstrip() + "\n"


def _presence_counts(units: list[KnowledgeUnit], keys: list[str]) -> dict[str, int]:
    return {
        key: sum(1 for unit in units if _has_metadata_value(unit.metadata, key))
        for key in keys
    }


def _has_metadata_value(metadata: Mapping[Any, Any], key: str) -> bool:
    return not _is_absent(metadata.get(key))


def _is_absent(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not _inline_text(value)
    if isinstance(value, list | dict):
        return not value
    return False


def _normalized_keys(keys: Sequence[str] | None) -> list[str]:
    if keys is None:
        return []
    return sorted({_inline_text(key) for key in keys if _inline_text(key)}, key=_text_sort_key)


def _unit_label(unit: KnowledgeUnit) -> str:
    unit_id = _inline_text(unit.id)
    title = _inline_text(unit.title)
    if unit_id and title:
        return f"{unit_id} ({title})"
    return unit_id or title or "Untitled"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (
        _inline_text(unit.id),
        _inline_text(unit.title),
        _inline_text(unit.source_id),
    )


def _text_sort_key(value: Any) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _coverage_percent(present: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{present / total * 100:.1f}%"


def _markdown_cell(value: Any) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")


def _inline_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()
