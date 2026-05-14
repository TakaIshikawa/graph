from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_metadata_namespace_summary_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def unit(unit_id: str, *, metadata: dict | None = None, source_project: object = SourceProject.MAX) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="Content",
        metadata=metadata or {},
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_metadata_namespace_summary_csv_empty_input_returns_header():
    assert export_metadata_namespace_summary_csv([]) == (
        "source_project,source_entity_type,namespace,key_count,unit_count,total_value_count,sample_keys\n"
    )


def test_export_metadata_namespace_summary_csv_groups_documented_namespaces():
    text = export_metadata_namespace_summary_csv(
        [
            unit(
                "a",
                metadata={
                    "source.url": "https://example.test",
                    "source_id": "1",
                    "review:state": "todo",
                    "review-owner": "me",
                    "plain": "root",
                },
            )
        ]
    )

    assert [row["namespace"] for row in rows(text)] == ["review", "root", "source"]


def test_export_metadata_namespace_summary_csv_counts_units_and_values():
    text = export_metadata_namespace_summary_csv(
        [
            unit("a", metadata={"source.url": "u", "source.tags": ["a", "b"]}),
            unit("b", metadata={"source.url": "v", "source.extra": {"x": 1, "y": 2}}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "max",
            "source_entity_type": "note",
            "namespace": "source",
            "key_count": "3",
            "unit_count": "2",
            "total_value_count": "6",
            "sample_keys": "source.extra; source.tags; source.url",
        }
    ]


def test_export_metadata_namespace_summary_csv_uses_unknown_fallbacks_and_path_mode(tmp_path):
    units = [unit("a", metadata={"x": 1}, source_project="")]
    path = tmp_path / "namespaces.csv"

    stats = export_metadata_namespace_summary_csv(units, path)

    assert rows(path.read_text(encoding="utf-8"))[0]["source_project"] == "Unknown"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
