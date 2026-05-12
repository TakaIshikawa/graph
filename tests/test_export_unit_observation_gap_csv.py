from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_observation_gap_csv import export_unit_observation_gap_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    title: str | None = None,
    source_project: SourceProject | str | None = SourceProject.MAX,
    source_entity_type: str | None = "note",
    created_at: object = None,
    ingested_at: object = None,
    updated_at: object = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=title or f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        created_at=created_at,
        ingested_at=ingested_at,
        updated_at=updated_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_observation_gap_csv_empty_input_has_header_only():
    assert export_unit_observation_gap_csv([]) == (
        "unit_id,unit_title,source_project,source_entity_type,observation_count,"
        "first_observed_date,last_observed_date,observed_span_days,largest_gap_days,"
        "has_multi_observation_gap\n"
    )


def test_unit_observation_gap_csv_reports_span_and_largest_adjacent_gap():
    text = export_unit_observation_gap_csv(
        [
            unit(
                "a",
                title="Alpha",
                source_project="Project A",
                source_entity_type="task",
                created_at="2024-01-01T09:00:00Z",
                updated_at="2024-01-10",
                metadata={
                    "observed_dates": ["2024-01-03", "2024-01-20", "not-a-date"],
                    "source_date": "2024-01-10",
                },
            )
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "unit_title": "Alpha",
            "source_project": "Project A",
            "source_entity_type": "task",
            "observation_count": "4",
            "first_observed_date": "2024-01-01",
            "last_observed_date": "2024-01-20",
            "observed_span_days": "19",
            "largest_gap_days": "10",
            "has_multi_observation_gap": "true",
        }
    ]


def test_unit_observation_gap_csv_includes_units_without_parsed_dates():
    text = export_unit_observation_gap_csv(
        [
            unit(
                "undated",
                title="Undated",
                source_project=None,
                source_entity_type=None,
                metadata={"observed_at": "unknown", "dates": ["also unknown"]},
            )
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "undated",
            "unit_title": "Undated",
            "source_project": "Unknown",
            "source_entity_type": "Unknown",
            "observation_count": "0",
            "first_observed_date": "",
            "last_observed_date": "",
            "observed_span_days": "",
            "largest_gap_days": "",
            "has_multi_observation_gap": "false",
        }
    ]


def test_unit_observation_gap_csv_sorts_by_unit_id_then_title():
    units = [
        unit("b", title="Beta", metadata={"date": "2024-02-01"}),
        unit("a", title="Zulu", metadata={"date": "2024-01-02"}),
        unit("a", title="Alpha", metadata={"date": "2024-01-01"}),
    ]

    assert export_unit_observation_gap_csv(units) == export_unit_observation_gap_csv(reversed(units))
    assert [(row["unit_id"], row["unit_title"]) for row in rows(export_unit_observation_gap_csv(units))] == [
        ("a", "Alpha"),
        ("a", "Zulu"),
        ("b", "Beta"),
    ]


def test_unit_observation_gap_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "unit-observation-gaps.csv"
    units = [unit("a", created_at="2024-01-01", metadata={"dates": ["2024-01-05"]})]

    expected = export_unit_observation_gap_csv(units)
    stats = export_unit_observation_gap_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
