"""Markdown export helpers for ChatGPT conversation digests."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_MISSING = "missing-conversation"


def export_units_to_chatgpt_digest_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    max_items_per_conversation: int = 5,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown digest of ChatGPT units."""
    text = _render(list(units), max_items_per_conversation=max_items_per_conversation)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {"path": str(output_path), "bytes_written": output_path.stat().st_size}


def _render(units: list[KnowledgeUnit], *, max_items_per_conversation: int) -> str:
    groups: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        if _source_project(unit) != "chatgpt_json" and "chatgpt" not in unit.tags:
            continue
        group = _text(unit.metadata.get("conversation_id") or unit.metadata.get("conversation") or unit.source_id) or _MISSING
        groups[group].append(unit)

    lines = ["# ChatGPT Conversation Digest", ""]
    for group_id, group_units in sorted(groups.items(), key=lambda item: (_first_created(item[1]).isoformat(), item[0])):
        ordered = sorted(group_units, key=lambda unit: (unit.created_at, unit.source_id))
        title = _text(ordered[0].metadata.get("title") or ordered[0].title) or "Untitled ChatGPT conversation"
        dates = [unit.created_at for unit in ordered]
        tags = Counter(tag for unit in ordered for tag in unit.tags if tag)
        lines.extend(
            [
                f"## {title}",
                "",
                f"- Conversation: {group_id}",
                f"- Date range: {min(dates).date().isoformat()} to {max(dates).date().isoformat()}",
                f"- Units: {len(ordered)}",
                f"- Top tags: {', '.join(tag for tag, _ in sorted(tags.items(), key=lambda item: (-item[1], item[0]))[:5])}",
                "",
            ]
        )
        for unit in ordered[:max_items_per_conversation]:
            lines.append(f"- {unit.created_at.isoformat()} - {_excerpt(unit.content or unit.title)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _first_created(units: list[KnowledgeUnit]):
    return min(unit.created_at for unit in units)


def _excerpt(value: str, *, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _source_project(unit: KnowledgeUnit) -> str:
    return _text(getattr(unit.source_project, "value", unit.source_project))


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())
