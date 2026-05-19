from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_collection_matrix_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, tags: list[str] | None = None, source_project: object = "Source", entity_type: str = "item") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=entity_type,
        title=f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        tags=tags or [],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_collection_matrix_supports_string_and_iterable_collections():
    text = export_source_collection_matrix_csv(
        [
            unit("b", metadata={"collection": "Inbox", "date": "2026-05-02"}),
            unit("a", metadata={"collections": ["Inbox", "Archive", "Inbox"], "date": "2026-05-01"}, tags=["x"]),
            unit("c", metadata={"folder": "Archive", "date": "2026-05-03"}),
            unit("d", metadata={}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Source",
            "collection": "Archive",
            "source_entity_type": "item",
            "unit_count": "2",
            "tagged_unit_count": "1",
            "first_seen": "2026-05-01",
            "last_seen": "2026-05-03",
            "representative_unit_ids": "a; c",
        },
        {
            "source_project": "Source",
            "collection": "Inbox",
            "source_entity_type": "item",
            "unit_count": "2",
            "tagged_unit_count": "1",
            "first_seen": "2026-05-01",
            "last_seen": "2026-05-02",
            "representative_unit_ids": "a; b",
        },
    ]


def test_source_collection_matrix_supports_mapping_inputs_and_unknowns():
    text = export_source_collection_matrix_csv([{"id": "a", "metadata": {"playlist": "Later"}, "tags": ["tag"]}])

    assert rows(text)[0]["source_project"] == "Unknown"
    assert rows(text)[0]["collection"] == "Later"
    assert rows(text)[0]["tagged_unit_count"] == "1"


def test_source_collection_matrix_path_writes(tmp_path):
    units = [unit("a", metadata={"project": "P"})]
    expected = export_source_collection_matrix_csv(units)
    path = tmp_path / "collections.csv"

    stats = export_source_collection_matrix_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
