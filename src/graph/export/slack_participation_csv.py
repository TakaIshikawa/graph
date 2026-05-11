"""CSV export helpers for Slack participation reports."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "channel",
    "user",
    "message_count",
    "first_message_at",
    "last_message_at",
    "thread_reply_count",
    "reaction_count",
    "top_titles",
]
_UNKNOWN = "unknown"


def export_units_to_slack_participation_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write Slack message participation summary rows."""
    rows = _summary_rows(list(units))
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "rows_written": len(rows)}


def _summary_rows(units: list[KnowledgeUnit]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "message_count": 0,
            "timestamps": [],
            "thread_reply_count": 0,
            "reaction_count": 0,
            "titles": [],
        }
    )
    for unit in units:
        if _source_project(unit) != "slack_json" or unit.source_entity_type != "slack_message":
            continue
        channel = _text(unit.metadata.get("channel") or unit.metadata.get("channel_name") or unit.metadata.get("channel_id")) or _UNKNOWN
        user = _text(unit.metadata.get("user") or unit.metadata.get("sender") or unit.metadata.get("username")) or _UNKNOWN
        row = grouped[(channel, user)]
        row["message_count"] += 1
        row["timestamps"].append(_text(unit.metadata.get("datetime")) or unit.created_at.isoformat())
        if unit.metadata.get("is_thread_reply") is True:
            row["thread_reply_count"] += 1
        row["reaction_count"] += _reaction_count(unit.metadata)
        if unit.title:
            row["titles"].append(unit.title)

    rows = []
    for (channel, user), summary in grouped.items():
        timestamps = sorted(timestamp for timestamp in summary["timestamps"] if timestamp)
        rows.append(
            {
                "channel": channel,
                "user": user,
                "message_count": summary["message_count"],
                "first_message_at": timestamps[0] if timestamps else "",
                "last_message_at": timestamps[-1] if timestamps else "",
                "thread_reply_count": summary["thread_reply_count"],
                "reaction_count": summary["reaction_count"],
                "top_titles": "; ".join(_unique(summary["titles"])[:3]),
            }
        )
    return sorted(rows, key=lambda row: (row["channel"], row["user"]))


def _reaction_count(metadata: dict[str, Any]) -> int:
    value = metadata.get("reaction_count")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    reactions = metadata.get("reactions")
    if not isinstance(reactions, list):
        return 0
    total = 0
    for reaction in reactions:
        if isinstance(reaction, dict):
            count = reaction.get("count")
            total += count if isinstance(count, int) and not isinstance(count, bool) else 0
    return total


def _render_csv(rows: list[dict[str, Any]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in sorted(values, key=lambda item: (item.casefold(), item)):
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _source_project(unit: KnowledgeUnit) -> str:
    return _text(getattr(unit.source_project, "value", unit.source_project))


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())
