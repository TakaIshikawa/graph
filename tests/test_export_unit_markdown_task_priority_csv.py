from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_markdown_task_priority_csv import export_units_to_markdown_task_priority_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_units_to_markdown_task_priority_csv_extracts_priority_markers():
    csv_text = export_units_to_markdown_task_priority_csv(
        [
            {
                "id": "u1",
                "content": "\n".join(
                    [
                        "- [ ] #priority/high Ship report",
                        "- [x] Call owner priority:: medium",
                        "- [ ] [priority: low] Groom backlog",
                        "- [ ] !!! Escalate outage",
                    ]
                ),
            }
        ]
    )

    rows = _rows(csv_text)
    assert [(row["line"], row["checked"], row["priority"], row["marker"], row["task_text"]) for row in rows] == [
        ("1", "false", "high", "#priority/high", "Ship report"),
        ("2", "true", "medium", "priority:: medium", "Call owner"),
        ("3", "false", "low", "[priority: low]", "Groom backlog"),
        ("4", "false", "high", "!!!", "Escalate outage"),
    ]


def test_export_units_to_markdown_task_priority_csv_ignores_fenced_code():
    csv_text = export_units_to_markdown_task_priority_csv(
        [{"id": "u1", "content": "```md\n- [ ] #priority/high Ignore\n```\n- [ ] !! Keep"}]
    )

    assert [(row["line"], row["priority"], row["task_text"]) for row in _rows(csv_text)] == [("4", "medium", "Keep")]


def test_export_units_to_markdown_task_priority_csv_writes_path(tmp_path):
    path = tmp_path / "tasks.csv"

    result = export_units_to_markdown_task_priority_csv([{"id": "u1", "content": "- [ ] ! Triage"}], path)

    assert result["path"] == str(path)
    assert result["unit_count"] == 1
    assert result["rows_exported"] == 1
    assert result["bytes_written"] == path.stat().st_size
