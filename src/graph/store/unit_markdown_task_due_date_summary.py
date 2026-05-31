"""Summarize Markdown task-list due dates."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_TASK_RE = re.compile(r"^\s*[-*+]\s+\[[ xX]\]\s+(?P<body>.*)$")
_DUE_RE = re.compile(r"(?:due::\s*|📅\s*|#due/)(\d{4}-\d{2}-\d{2})")


def summarize_unit_markdown_task_due_dates(units: Iterable[Any], as_of: str | date | None = None, sample_limit: int = 5) -> dict[str, Any]:
    cutoff = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    total = with_due = overdue = 0
    counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for unit in units:
        for line_number, body in _tasks(str(get(unit, "content") or "")):
            total += 1
            dates = sorted(set(_DUE_RE.findall(body)))
            if not dates:
                continue
            with_due += 1
            for value in dates:
                counts[value] += 1
                if cutoff and date.fromisoformat(value) < cutoff:
                    overdue += 1
            if len(samples) < sample_limit:
                samples.append({"unit_id": unit_id(unit), "line": line_number, "due_dates": dates, "text": field_value(body)})
    return {"total_tasks": total, "tasks_with_due_dates": with_due, "overdue_count": overdue, "date_counts": [{"date": key, "count": counts[key]} for key in sorted(counts)], "samples": samples}


def _tasks(content: str) -> list[tuple[int, str]]:
    return [(line_number, match.group("body")) for line_number, line in enumerate(content.splitlines(), start=1) if (match := _TASK_RE.match(line))]
