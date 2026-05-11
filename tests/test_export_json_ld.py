from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.export import export_units_to_json_ld
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
UPDATED_TIME = datetime(2026, 5, 2, 8, 30, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    title: str = "Research note",
    *,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.CSV,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content="A compact research note.",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags or ["schema", "export"],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UPDATED_TIME,
    )


def test_export_units_to_json_ld_emits_single_creative_work_node():
    data = json.loads(
        export_units_to_json_ld(
            unit(
                "unit a",
                metadata={
                    "url": "https://example.test/notes/unit-a",
                    "same_as": ["https://archive.example.test/a"],
                    "rating": 5,
                    "nested": {"b": 2, "a": 1},
                },
            )
        )
    )

    assert data == {
        "@context": "https://schema.org",
        "@id": "urn:knowledge-unit:unit%20a",
        "@type": "CreativeWork",
        "additionalProperty": [
            {"@type": "PropertyValue", "name": "nested", "value": {"a": 1, "b": 2}},
            {"@type": "PropertyValue", "name": "rating", "value": 5},
            {"@type": "PropertyValue", "name": "same_as", "value": ["https://archive.example.test/a"]},
            {"@type": "PropertyValue", "name": "url", "value": "https://example.test/notes/unit-a"},
        ],
        "dateCreated": "2026-05-01T10:15:00+00:00",
        "dateModified": "2026-05-02T08:30:00+00:00",
        "encodingFormat": "insight",
        "isBasedOn": {
            "@type": "CreativeWork",
            "additionalType": "note",
            "identifier": "source-unit a",
            "name": "csv",
        },
        "keywords": ["export", "schema"],
        "name": "Research note",
        "sameAs": "https://archive.example.test/a",
        "text": "A compact research note.",
        "url": "https://example.test/notes/unit-a",
    }


def test_export_units_to_json_ld_emits_deterministic_graph_for_multiple_units():
    text_a = export_units_to_json_ld(
        [
            unit("unit-b", "Beta", tags=["zeta"]),
            unit("unit-a", "Alpha", metadata={"source_url": "https://example.test/a"}, tags=["alpha"]),
        ]
    )
    text_b = export_units_to_json_ld(
        [
            unit("unit-a", "Alpha", metadata={"source_url": "https://example.test/a"}, tags=["alpha"]),
            unit("unit-b", "Beta", tags=["zeta"]),
        ]
    )
    data = json.loads(text_a)

    assert text_a == text_b
    assert data["@context"] == "https://schema.org"
    assert [node["@id"] for node in data["@graph"]] == ["urn:knowledge-unit:unit-a", "urn:knowledge-unit:unit-b"]
    assert [node["name"] for node in data["@graph"]] == ["Alpha", "Beta"]
    assert data["@graph"][0]["url"] == "https://example.test/a"


def test_export_units_to_json_ld_writes_file(tmp_path):
    path = tmp_path / "nested" / "units.jsonld"

    text = export_units_to_json_ld(unit("unit-a"), path)

    assert path.read_text(encoding="utf-8") == text
