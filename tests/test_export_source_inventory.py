from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_source_inventory_csv
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(
    source_project: SourceProject | str,
    source_id: str,
    source_entity_type: str,
    created_at: datetime,
    tags: list[str] | None = None,
    unit_id: str = "",
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=source_id,
        source_entity_type=source_entity_type,
        title=source_id,
        content="",
        content_type=ContentType.METADATA,
        tags=tags or [],
        created_at=created_at,
        updated_at=created_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_inventory_groups_sources_types_tags_and_edges():
    created = datetime(2025, 1, 1, tzinfo=timezone.utc)
    units = [
        unit(SourceProject.MAX, "max-1", "note", created, ["alpha"], unit_id="u1"),
        unit(SourceProject.MAX, "max-2", "task", datetime(2025, 1, 2, tzinfo=timezone.utc), ["alpha", "beta"], unit_id="u2"),
        unit(SourceProject.PINBOARD, "pin-1", "bookmark", datetime(2025, 1, 3, tzinfo=timezone.utc), ["beta"], unit_id="u3"),
    ]
    edges = [KnowledgeEdge(from_unit_id="u1", to_unit_id="u2", relation=EdgeRelation.RELATES_TO)]

    output = export_source_inventory_csv(units, edges)
    parsed = rows(output)

    assert [row["source_project"] for row in parsed] == ["max", "pinboard"]
    assert parsed[0]["unit_count"] == "2"
    assert parsed[0]["edge_count"] == "1"
    assert parsed[0]["first_timestamp"] == "2025-01-01T00:00:00+00:00"
    assert parsed[0]["last_timestamp"] == "2025-01-02T00:00:00+00:00"
    assert parsed[0]["types"] == "note:1;task:1"
    assert parsed[0]["tags"] == "alpha:2;beta:1"


def test_source_inventory_uses_unknown_fallback_and_deterministic_order():
    created = datetime(2025, 1, 1, tzinfo=timezone.utc)
    units = [
        unit("zeta", "z", "", created),
        unit("", "missing", "note", created),
        unit("alpha", "a", "note", created),
    ]

    first = export_source_inventory_csv(units)
    second = export_source_inventory_csv(reversed(units))

    assert first == second
    parsed = rows(first)
    assert [row["source_project"] for row in parsed] == ["alpha", "unknown", "zeta"]
    assert parsed[1]["types"] == "note:1"


def test_source_inventory_path_mode_writes_file(tmp_path):
    output_path = tmp_path / "source_inventory.csv"
    stats = export_source_inventory_csv(
        [unit(SourceProject.MAX, "max-1", "note", datetime(2025, 1, 1, tzinfo=timezone.utc))],
        path=output_path,
    )

    assert stats["rows_written"] == 1
    assert output_path.read_text(encoding="utf-8").startswith("source,source_project")
