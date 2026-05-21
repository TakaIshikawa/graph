from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_units_to_metadata_namespace_matrix_csv
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def unit(unit_id: str, *, metadata: dict | None = None, source_project: object = "max") -> KnowledgeUnit:
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


def test_export_units_to_metadata_namespace_matrix_csv_empty_input_returns_header():
    assert export_units_to_metadata_namespace_matrix_csv([]) == (
        "unit_id,title,source_project,namespace,key_count,populated_value_count,empty_value_count,list_value_count,scalar_value_count,keys\n"
    )


def test_export_units_to_metadata_namespace_matrix_csv_groups_namespaces_per_unit():
    text = export_units_to_metadata_namespace_matrix_csv(
        [
            unit(
                "b",
                metadata={
                    "source.url": "https://example.test",
                    "source_id": "1",
                    "review:state": "",
                    "file/path": ["a.pdf", "b.pdf"],
                    "plain": "root",
                },
            )
        ]
    )

    result = rows(text)
    assert [row["namespace"] for row in result] == ["file", "review", "source", "unscoped"]
    assert result[2]["key_count"] == "2"
    assert result[2]["keys"] == "source.url; source_id"
    assert result[1]["empty_value_count"] == "1"
    assert result[0]["list_value_count"] == "1"


def test_export_units_to_metadata_namespace_matrix_csv_counts_nested_list_and_scalar_values():
    result = rows(
        export_units_to_metadata_namespace_matrix_csv(
            [
                {
                    "id": "a",
                    "title": "A",
                    "source_project": "",
                    "metadata": {
                        "meta.tags": ["x"],
                        "meta.empty_list": [],
                        "meta.details": {"owner": "me"},
                        "meta.none": None,
                    },
                }
            ]
        )
    )[0]

    assert result["source_project"] == "Unknown"
    assert result["namespace"] == "meta"
    assert result["key_count"] == "4"
    assert result["populated_value_count"] == "2"
    assert result["empty_value_count"] == "2"
    assert result["list_value_count"] == "2"
    assert result["scalar_value_count"] == "2"


def test_export_units_to_metadata_namespace_matrix_csv_sorts_by_unit_title_namespace_and_writes_path(tmp_path):
    path = tmp_path / "matrix.csv"

    stats = export_units_to_metadata_namespace_matrix_csv(
        [
            {"id": "b", "title": "B", "metadata": {"z.key": 1}},
            {"id": "a", "title": "A", "metadata": {"z.key": 1}},
        ],
        path,
    )

    assert [row["unit_id"] for row in rows(path.read_text(encoding="utf-8"))] == ["a", "b"]
    assert stats["unit_count"] == 2
    assert stats["rows_exported"] == 2
