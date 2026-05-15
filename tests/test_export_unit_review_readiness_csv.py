from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_unit_review_readiness_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def unit(unit_id: str, **overrides: object) -> KnowledgeUnit:
    values = {
        "id": unit_id,
        "source_project": SourceProject.MAX,
        "source_id": f"source-{unit_id}",
        "source_entity_type": "note",
        "title": f"Title {unit_id}",
        "content": "Content",
        "metadata": {},
        "tags": ["tag"],
        "created_at": UNIT_TIME,
        "ingested_at": UNIT_TIME,
        "updated_at": UNIT_TIME,
    }
    values.update(overrides)
    return KnowledgeUnit.model_construct(**values)


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_review_readiness_csv_empty_input_returns_header():
    assert export_unit_review_readiness_csv([]) == (
        "unit_id,title,source_project,source_entity_type,missing_field_count,missing_fields,"
        "review_status,needs_review\n"
    )


def test_export_unit_review_readiness_csv_emits_one_row_per_unit():
    text = export_unit_review_readiness_csv([unit("b"), unit("a")])

    assert [row["unit_id"] for row in rows(text)] == ["a", "b"]


def test_export_unit_review_readiness_csv_reports_missing_fields_stably():
    text = export_unit_review_readiness_csv(
        [
            unit(
                "missing",
                title=" ",
                content="",
                tags=[" "],
                source_id="",
                source_project="",
                source_entity_type=None,
            )
        ]
    )

    row = rows(text)[0]
    assert row["missing_field_count"] == "6"
    assert row["missing_fields"] == "title;content;tags;source_id;source_project;source_entity_type"
    assert row["needs_review"] == "true"


def test_export_unit_review_readiness_csv_normalizes_review_metadata():
    text = export_unit_review_readiness_csv(
        [
            unit("ready", metadata={"reviewed": True}),
            unit("triage", metadata={"triage_status": "Pending"}),
            unit("explicit", metadata={"needs_review": "yes", "status": "approved"}),
        ]
    )

    result = {row["unit_id"]: row for row in rows(text)}
    assert result["ready"]["review_status"] == "reviewed"
    assert result["ready"]["needs_review"] == "false"
    assert result["triage"]["review_status"] == "pending"
    assert result["triage"]["needs_review"] == "true"
    assert result["explicit"]["review_status"] == "approved"
    assert result["explicit"]["needs_review"] == "true"


def test_export_unit_review_readiness_csv_path_mode(tmp_path):
    path = tmp_path / "readiness.csv"
    stats = export_unit_review_readiness_csv([unit("a")], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["unit_id"] == "a"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
