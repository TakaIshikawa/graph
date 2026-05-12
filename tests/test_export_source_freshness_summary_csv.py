from __future__ import annotations

import csv
from datetime import datetime
from io import StringIO

from graph.export import export_source_freshness_summary_csv
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(
    unit_id: str,
    source_project: SourceProject | str | None,
    source_entity_type: str | None,
    *,
    created_at: object = None,
    updated_at: object = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        created_at=created_at,
        ingested_at=None,
        updated_at=updated_at,
    )


def edge(edge_id: str, metadata: dict) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id="from",
        to_unit_id="to",
        relation=EdgeRelation.REFERENCES,
        source=EdgeSource.SOURCE,
        weight=1.0,
        metadata=metadata,
        created_at=datetime(2024, 2, 1, 12, 0, 0),
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_freshness_summary_empty_input_has_header_only():
    assert export_source_freshness_summary_csv([]) == (
        "source_project,source_entity_type,first_seen_date,last_seen_date,"
        "observed_date_span_days,unit_count,edge_count\n"
    )


def test_source_freshness_summary_handles_missing_dates():
    text = export_source_freshness_summary_csv([unit("a", None, None)])

    assert rows(text) == [
        {
            "source_project": "Unknown",
            "source_entity_type": "Unknown",
            "first_seen_date": "",
            "last_seen_date": "",
            "observed_date_span_days": "",
            "unit_count": "1",
            "edge_count": "0",
        }
    ]


def test_source_freshness_summary_groups_and_sorts_multiple_sources():
    text = export_source_freshness_summary_csv(
        [
            unit("a", "Source B", "note", created_at="2024-04-10", updated_at="2024-04-12"),
            unit("b", SourceProject.MAX, "note", created_at="2024-01-01", updated_at="2024-01-05"),
            unit("c", SourceProject.MAX, "bookmark", metadata={"observed_date": "2024-03-15"}),
        ],
        edges=[
            edge(
                "e1",
                {
                    "source_project": SourceProject.MAX,
                    "source_entity_type": "note",
                    "observed_at": "2024-01-10T09:00:00Z",
                },
            )
        ],
    )

    assert rows(text) == [
        {
            "source_project": "max",
            "source_entity_type": "bookmark",
            "first_seen_date": "2024-03-15",
            "last_seen_date": "2024-03-15",
            "observed_date_span_days": "0",
            "unit_count": "1",
            "edge_count": "0",
        },
        {
            "source_project": "max",
            "source_entity_type": "note",
            "first_seen_date": "2024-01-01",
            "last_seen_date": "2024-01-10",
            "observed_date_span_days": "9",
            "unit_count": "1",
            "edge_count": "1",
        },
        {
            "source_project": "Source B",
            "source_entity_type": "note",
            "first_seen_date": "2024-04-10",
            "last_seen_date": "2024-04-12",
            "observed_date_span_days": "2",
            "unit_count": "1",
            "edge_count": "0",
        },
    ]


def test_source_freshness_summary_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "freshness.csv"
    units = [unit("a", "Source A", "note", created_at="2024-01-01")]

    expected = export_source_freshness_summary_csv(units)
    stats = export_source_freshness_summary_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "edge_count": 0,
        "source_type_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
