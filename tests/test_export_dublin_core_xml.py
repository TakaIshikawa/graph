from __future__ import annotations

from datetime import datetime, timezone
from xml.etree import ElementTree as ET

from graph.export import export_units_to_dublin_core_xml
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
DC = "{http://purl.org/dc/elements/1.1/}"


def unit(metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id="unit-a",
        source_project=SourceProject.BIBTEX,
        source_id="source-a",
        source_entity_type="publication",
        title="Graph Exports",
        content="Fallback description.",
        content_type=ContentType.FINDING,
        tags=["graphs", "exports"],
        metadata=metadata or {},
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def test_export_units_to_dublin_core_xml_maps_common_fields():
    root = ET.fromstring(
        export_units_to_dublin_core_xml(
            [
                unit(
                    {
                        "authors": ["Ada Smith", {"name": "Grace Doe"}],
                        "abstract": "A graph export paper.",
                        "publisher": "Example Press",
                        "published_at": "2025-04-24",
                        "doi": "10.1234/example",
                        "url": "https://example.test",
                        "language": "en",
                        "relation": ["unit-b"],
                    }
                )
            ]
        )
    )
    record = root.find("record")
    assert record is not None
    assert [item.text for item in record.findall(f"{DC}creator")] == ["Ada Smith", "Grace Doe"]
    assert [item.text for item in record.findall(f"{DC}subject")] == ["exports", "graphs"]
    assert [item.text for item in record.findall(f"{DC}identifier")] == [
        "source-a",
        "10.1234/example",
        "https://example.test",
    ]
    assert record.findtext(f"{DC}description") == "A graph export paper."


def test_export_units_to_dublin_core_xml_writes_path(tmp_path):
    path = tmp_path / "dc.xml"

    stats = export_units_to_dublin_core_xml([unit()], path)

    ET.fromstring(path.read_text(encoding="utf-8"))
    assert stats["bytes_written"] == path.stat().st_size
