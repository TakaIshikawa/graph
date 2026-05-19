from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_metadata_conflicts_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, source_project: object = "Source") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_metadata_conflicts_reports_conflicting_synonym_groups():
    text = export_unit_metadata_conflicts_csv(
        [
            unit("a", metadata={"url": "https://example.com/a", "source_url": "https://example.com/b"}),
            unit("b", metadata={"account": "Checking", "account_name": "Savings"}),
            unit("c", metadata={"title": "Same", "name": "Same"}),
        ]
    )

    assert [(row["unit_id"], row["conflict_group"], row["keys_present"]) for row in rows(text)] == [
        ("a", "url", "source_url; url"),
        ("b", "account", "account; account_name"),
    ]


def test_unit_metadata_conflicts_normalizes_equivalent_dates_and_amounts():
    text = export_unit_metadata_conflicts_csv(
        [
            unit("a", metadata={"date": "2026-05-01", "published_at": "2026-05-01T12:00:00Z"}),
            unit("b", metadata={"amount": "$1,200.00", "net_amount": "1200"}),
            unit("c", metadata={"amount": "$1,200.00", "net_amount": "1201"}),
        ]
    )

    assert [(row["unit_id"], row["conflict_group"], row["normalized_values"]) for row in rows(text)] == [
        ("c", "amount", "1200; 1201")
    ]


def test_unit_metadata_conflicts_supports_mapping_inputs_and_path_writes(tmp_path):
    units = [{"id": "a", "metadata": {"title": "One", "description": "Two"}}]
    expected = export_unit_metadata_conflicts_csv(units)
    path = tmp_path / "conflicts.csv"
    stats = export_unit_metadata_conflicts_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert rows(expected)[0]["source_project"] == "Unknown"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
