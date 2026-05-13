from __future__ import annotations

import csv
from io import StringIO

from graph.export.source_activity_cadence_csv import export_source_activity_cadence_csv
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: str = "Project A",
    source_entity_type: str = "note",
    metadata: dict | None = None,
    created_at: object = None,
    updated_at: object = None,
    ingested_at: object = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=[],
        created_at=created_at,
        updated_at=updated_at,
        ingested_at=ingested_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_activity_cadence_csv_empty_input_has_header_only():
    assert export_source_activity_cadence_csv([]) == (
        "source_project,source_entity_type,observation_count,first_seen_date,last_seen_date,"
        "active_span_days,average_gap_days,max_gap_days\n"
    )


def test_source_activity_cadence_csv_groups_dates_and_calculates_gaps():
    text = export_source_activity_cadence_csv(
        [
            unit("c", metadata={"date": "2024-01-11"}),
            unit("a", metadata={"date": "2024-01-01T08:00:00Z"}),
            unit("b", metadata={"published_at": "2024-01-04"}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Project A",
            "source_entity_type": "note",
            "observation_count": "3",
            "first_seen_date": "2024-01-01",
            "last_seen_date": "2024-01-11",
            "active_span_days": "10",
            "average_gap_days": "5",
            "max_gap_days": "7",
        }
    ]


def test_source_activity_cadence_csv_invalid_or_missing_dates_leave_blank_cadence():
    text = export_source_activity_cadence_csv(
        [
            unit("a", metadata={"date": "not-a-date"}),
            unit("b", metadata={}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Project A",
            "source_entity_type": "note",
            "observation_count": "2",
            "first_seen_date": "",
            "last_seen_date": "",
            "active_span_days": "",
            "average_gap_days": "",
            "max_gap_days": "",
        }
    ]


def test_source_activity_cadence_csv_sorts_groups_deterministically():
    text = export_source_activity_cadence_csv(
        [
            unit("b", source_project="Project B", source_entity_type="task", metadata={"date": "2024-01-02"}),
            unit("a", source_project="Project A", source_entity_type="note", metadata={"date": "2024-01-01"}),
        ]
    )

    assert [row["source_project"] for row in rows(text)] == ["Project A", "Project B"]


def test_source_activity_cadence_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "source-activity-cadence.csv"
    units = [unit("a", metadata={"date": "2024-01-01"})]

    expected = export_source_activity_cadence_csv(units)
    stats = export_source_activity_cadence_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "group_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
