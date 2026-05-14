from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_unit_reading_queue_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    content: str = "ordinary content",
    metadata: dict | None = None,
    confidence: float | None = 0.9,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content=content,
        metadata=metadata or {},
        tags=[],
        confidence=confidence,
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_reading_queue_csv_empty_input_returns_header():
    assert export_unit_reading_queue_csv([]) == (
        "unit_id,title,source_project,source_entity_type,priority_score,reason_codes,"
        "estimated_reading_minutes,best_date\n"
    )


def test_export_unit_reading_queue_csv_prioritizes_read_later_units():
    text = export_unit_reading_queue_csv(
        [
            unit("ordinary", metadata={}),
            unit("later", metadata={"read_later": True}),
        ]
    )

    result = rows(text)
    assert result[0]["unit_id"] == "later"
    assert int(result[0]["priority_score"]) > int(result[1]["priority_score"])
    assert "read_later" in result[0]["reason_codes"]


def test_export_unit_reading_queue_csv_estimates_reading_minutes_with_minimum():
    text = export_unit_reading_queue_csv(
        [
            unit("short", content="word"),
            unit("empty", content=""),
            unit("long", content=" ".join(["word"] * 401)),
        ]
    )

    minutes = {row["unit_id"]: row["estimated_reading_minutes"] for row in rows(text)}
    assert minutes["short"] == "1"
    assert minutes["empty"] == "0"
    assert minutes["long"] == "3"


def test_export_unit_reading_queue_csv_uses_metadata_best_date():
    text = export_unit_reading_queue_csv([unit("a", metadata={"published_at": "2026-06-01"})])

    assert rows(text)[0]["best_date"] == "2026-06-01"


def test_export_unit_reading_queue_csv_path_mode(tmp_path):
    units = [unit("a", metadata={"unread": "true"})]
    path = tmp_path / "queue.csv"

    stats = export_unit_reading_queue_csv(units, path)

    assert rows(path.read_text(encoding="utf-8"))[0]["unit_id"] == "a"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
