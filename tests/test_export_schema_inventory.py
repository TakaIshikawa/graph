from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_unit_schema_inventory
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    content_type: ContentType = ContentType.INSIGHT,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=content_type,
        tags=tags or [],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
        metadata=metadata or {},
    )


def test_schema_inventory_flattens_nested_metadata_with_dotted_paths():
    report = export_unit_schema_inventory(
        [
            unit(
                "unit-a",
                metadata={
                    "paper": {
                        "doi": "10.123/example",
                        "authors": {"count": 2},
                    },
                    "read_status": "unread",
                },
            )
        ]
    )

    assert "| paper.authors.count | 1 | integer (1) | 2 |" in report
    assert "| paper.doi | 1 | string (1) | 10.123/example |" in report
    assert "| read_status | 1 | string (1) | unread |" in report


def test_schema_inventory_counts_types_and_occurrences():
    report = export_unit_schema_inventory(
        [
            unit("unit-a", metadata={"priority": "high", "score": 3, "flagged": True}),
            unit("unit-b", metadata={"priority": 2, "score": 3.5, "missing": None}),
            unit("unit-c", metadata={"priority": "low", "score": 7}),
        ]
    )

    assert "| priority | 3 | string (2), integer (1) | 2; high; low |" in report
    assert "| score | 3 | integer (2), number (1) | 3; 3.5; 7 |" in report
    assert "| boolean | 1 |" in report
    assert "| null | 1 |" in report


def test_schema_inventory_orders_sections_deterministically_and_sanitizes_examples():
    units = [
        unit(
            "unit-c",
            source_project=SourceProject.PINBOARD,
            content_type=ContentType.FINDING,
            metadata={"note": "zeta\nwith | pipe"},
            tags=["beta", "alpha"],
        ),
        unit(
            "unit-a",
            source_project=SourceProject.MAX,
            content_type=ContentType.INSIGHT,
            metadata={"note": "alpha", "long": "x" * 120},
            tags=["beta"],
        ),
    ]

    first = export_unit_schema_inventory(units)
    second = export_unit_schema_inventory(reversed(units))

    assert first == second
    assert "| note | 2 | string (2) | alpha; zeta with \\| pipe |" in first
    assert "| long | 1 | string (1) | " + ("x" * 79) + "... |" in first
    assert ("x" * 120) not in first
    assert "| beta | 2 |" in first
    assert "| alpha | 1 |" in first
    assert "| max | 1 |" in first
    assert "| pinboard | 1 |" in first
    assert "| finding | 1 |" in first
    assert "| insight | 1 |" in first


def test_schema_inventory_empty_input_returns_valid_zero_count_report():
    report = export_unit_schema_inventory([])

    assert "# Unit Schema Inventory" in report
    assert "| Units scanned | 0 |" in report
    assert "| Metadata keys | 0 |" in report
    assert "| _None_ | 0 |" in report


def test_schema_inventory_path_writes_returned_report(tmp_path):
    path = tmp_path / "reports" / "schema.md"

    report = export_unit_schema_inventory([unit("unit-a", metadata={"priority": "high"})], path)

    assert path.read_text(encoding="utf-8") == report
    assert "| priority | 1 | string (1) | high |" in report
