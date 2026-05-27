"""CSV export for inline hashtags in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "hashtag", "normalized_hashtag", "line_number", "occurrence_count"]
_HASHTAG_RE = re.compile(r"(?<![\w`])#([A-Za-z0-9][A-Za-z0-9_-]*)")


def export_units_to_hashtag_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        counts: Counter[tuple[str, str, int]] = Counter()
        in_fence = False
        for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence or line.lstrip().startswith("#"):
                continue
            for match in _HASHTAG_RE.finditer(line):
                tag = f"#{match.group(1)}"
                counts[(tag, tag.casefold(), line_number)] += 1
        for (tag, normalized, line_number), count in counts.items():
            rows.append({"unit_id": unit_id(unit), "title": title, "hashtag": tag, "normalized_hashtag": normalized, "line_number": line_number, "occurrence_count": count})
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["normalized_hashtag"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}
