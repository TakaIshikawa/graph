from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export.unit_empty_content_csv import export_unit_empty_content_csv
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    title: str,
    content: str | None,
    *,
    source_project: str = "Project A",
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=content,
        metadata=metadata or {},
        tags=tags or [],
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_empty_content_csv_empty_input_has_header_only():
    assert export_unit_empty_content_csv([]) == (
        "unit_id,title,source_project,source_entity_type,content_state,metadata_key_count,"
        "tag_count,created_at,updated_at\n"
    )


def test_unit_empty_content_csv_reports_missing_empty_and_whitespace_only_content():
    text = export_unit_empty_content_csv(
        [
            unit("ok", "Filled", "content"),
            unit("missing", "Missing", None, metadata={"url": "x"}, tags=["repair"]),
            unit("empty", "Empty", ""),
            unit("space", "Whitespace", " \n\t "),
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "empty",
            "title": "Empty",
            "source_project": "Project A",
            "source_entity_type": "note",
            "content_state": "empty",
            "metadata_key_count": "0",
            "tag_count": "0",
            "created_at": "2024-01-01 00:00:00+00:00",
            "updated_at": "2024-01-02 00:00:00+00:00",
        },
        {
            "unit_id": "missing",
            "title": "Missing",
            "source_project": "Project A",
            "source_entity_type": "note",
            "content_state": "missing",
            "metadata_key_count": "1",
            "tag_count": "1",
            "created_at": "2024-01-01 00:00:00+00:00",
            "updated_at": "2024-01-02 00:00:00+00:00",
        },
        {
            "unit_id": "space",
            "title": "Whitespace",
            "source_project": "Project A",
            "source_entity_type": "note",
            "content_state": "whitespace",
            "metadata_key_count": "0",
            "tag_count": "0",
            "created_at": "2024-01-01 00:00:00+00:00",
            "updated_at": "2024-01-02 00:00:00+00:00",
        },
    ]


def test_unit_empty_content_csv_sorts_by_source_title_unit_id_and_writes_path(tmp_path):
    units = [
        {"id": "b", "title": "Beta", "source_project": "Project B", "source_entity_type": "doc", "content": ""},
        {"id": "a2", "title": "Alpha", "source_project": "Project A", "content": ""},
        {"id": "a1", "title": "Alpha", "source_project": "Project A", "content": None, "tags": ["x", "y"]},
    ]
    path = tmp_path / "reports" / "empty.csv"

    expected = export_unit_empty_content_csv(units)
    stats = export_unit_empty_content_csv(units, path)

    assert [row["unit_id"] for row in rows(expected)] == ["a1", "a2", "b"]
    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 3,
        "empty_content_count": 3,
        "rows_exported": 3,
        "bytes_written": path.stat().st_size,
    }
