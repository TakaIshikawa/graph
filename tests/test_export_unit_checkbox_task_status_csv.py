from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_checkbox_task_status_csv import export_units_to_checkbox_task_status_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_checkbox_task_status_counts_mixed_markers():
    text = export_units_to_checkbox_task_status_csv(
        [
            {
                "id": "u1",
                "content": "\n".join(
                    [
                        "- [x] Done",
                        "- [X] Also done",
                        "- [ ] Open",
                        "- [-] Canceled",
                        "- [~] Deferred",
                    ]
                ),
            }
        ]
    )

    assert _rows(text) == [
        {
            "unit_id": "u1",
            "task_count": "5",
            "completed_count": "2",
            "open_count": "1",
            "canceled_count": "2",
            "completion_ratio": "0.40",
            "first_open_task": "Open",
        }
    ]


def test_checkbox_task_status_includes_no_task_units_with_empty_ratio():
    rows = _rows(export_units_to_checkbox_task_status_csv([{"id": "empty", "content": "No tasks here."}]))

    assert rows[0]["task_count"] == "0"
    assert rows[0]["completion_ratio"] == ""
    assert rows[0]["first_open_task"] == ""


def test_checkbox_task_status_trims_first_open_task():
    rows = _rows(export_units_to_checkbox_task_status_csv([{"unit_id": "u1", "content": "- [ ]   Write   the   report  "}]))

    assert rows[0]["first_open_task"] == "Write the report"


def test_checkbox_task_status_numeric_format_is_deterministic():
    rows = _rows(export_units_to_checkbox_task_status_csv([{"source_id": "u1", "content": "- [x] A\n- [ ] B\n- [ ] C"}]))

    assert rows[0]["completion_ratio"] == "0.33"


def test_checkbox_task_status_writes_path(tmp_path):
    path = tmp_path / "reports" / "tasks.csv"

    result = export_units_to_checkbox_task_status_csv([{"id": "u1", "content": "- [ ] Ship"}], path)

    assert result["path"] == str(path)
    assert result["rows_exported"] == 1
    assert _rows(path.read_text(encoding="utf-8"))[0]["open_count"] == "1"
