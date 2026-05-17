from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_title_quality_csv import export_unit_title_quality_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, title: str | None, source_project: str = "Project A") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content="content",
        metadata={},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_title_quality_csv_empty_input_has_header_only():
    assert export_unit_title_quality_csv([]) == (
        "unit_id,title,source_project,issue_count,issues,title_length,duplicate_count\n"
    )


def test_unit_title_quality_csv_reports_requested_issue_types_only():
    long_title = "L" * 121
    text = export_unit_title_quality_csv(
        [
            unit("good", "A strong useful title"),
            unit("missing", ""),
            unit("generic", "Untitled"),
            unit("dup-1", "Duplicate Title"),
            unit("dup-2", " duplicate   title "),
            unit("long", long_title),
            unit("url", "https://example.test/item"),
            unit("same-as-id", "same-as-id"),
        ]
    )

    by_id = {row["unit_id"]: row for row in rows(text)}
    assert set(by_id) == {"missing", "generic", "dup-1", "dup-2", "long", "url", "same-as-id"}
    assert by_id["missing"]["issues"] == "missing"
    assert by_id["generic"]["issues"] == "generic"
    assert by_id["dup-1"]["issues"] == "duplicate"
    assert by_id["dup-1"]["duplicate_count"] == "2"
    assert by_id["dup-2"]["issues"] == "duplicate"
    assert by_id["long"]["issues"] == "long"
    assert by_id["long"]["title_length"] == "121"
    assert by_id["url"]["issues"] == "url_like"
    assert by_id["same-as-id"]["issues"] == "same_as_id"


def test_unit_title_quality_csv_combines_issues_accepts_mappings_and_writes_path(tmp_path):
    units = [
        {"id": "note", "title": "Note", "source_project": "Project B"},
        {"id": "also-note", "title": " note ", "source_project": "Project B"},
    ]
    path = tmp_path / "reports" / "titles.csv"

    expected = export_unit_title_quality_csv(units)
    stats = export_unit_title_quality_csv(units, path)

    assert rows(expected) == [
        {
            "unit_id": "note",
            "title": "Note",
            "source_project": "Project B",
            "issue_count": "3",
            "issues": "generic; duplicate; same_as_id",
            "title_length": "4",
            "duplicate_count": "2",
        },
        {
            "unit_id": "also-note",
            "title": "note",
            "source_project": "Project B",
            "issue_count": "2",
            "issues": "generic; duplicate",
            "title_length": "4",
            "duplicate_count": "2",
        },
    ]
    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "weak_title_count": 2,
        "rows_exported": 2,
        "bytes_written": path.stat().st_size,
    }
