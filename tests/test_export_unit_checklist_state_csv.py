from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_checklist_state_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_checklist_state_csv_classifies_markers_and_indent():
    result = rows(export_units_to_checklist_state_csv([{"id": "u", "title": "Tasks", "content": "- [ ] Open\n  - [x] Done\n    - [?] Custom"}]))

    assert [row["state"] for row in result] == ["open", "done", "custom"]
    assert [row["indentation_depth"] for row in result] == ["0", "1", "2"]
    assert result[2]["state_marker"] == "?"


def test_checklist_state_csv_ignores_fenced_code():
    result = rows(export_units_to_checklist_state_csv([{"id": "u", "content": "```\n- [ ] Ignore\n```\n- [ ] Keep"}]))

    assert len(result) == 1
    assert result[0]["item_text"] == "Keep"
