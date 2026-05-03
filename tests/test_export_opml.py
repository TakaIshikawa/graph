from __future__ import annotations

from datetime import datetime, timezone
from xml.etree import ElementTree as ET

from graph.export import export_units_to_opml
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
UPDATED_TIME = datetime(2026, 5, 2, 8, 30, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    title: str,
    tags: list[str] | None = None,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    source_id: str | None = None,
    content_type: ContentType | str = ContentType.INSIGHT,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=source_id or f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} content",
        content_type=content_type,
        metadata={},
        tags=tags or [],
        confidence=0.8,
        utility_score=0.6,
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UPDATED_TIME,
    )


def parsed_body(xml: str) -> ET.Element:
    root = ET.fromstring(xml)
    assert root.tag == "opml"
    assert root.attrib["version"] == "2.0"
    body = root.find("body")
    assert body is not None, "OPML must have a body element"
    return body


def outline_texts(outline: ET.Element) -> list[str]:
    return [item.attrib["text"] for item in outline.findall("outline")]


def test_export_units_to_opml_escapes_xml_and_includes_unit_attributes():
    xml = export_units_to_opml(
        [
            unit(
                "unit-a",
                'A&B <Plan> "quoted"',
                ['tag & <one> "q"'],
                source_project="source & project",
                source_id='source<&>"',
            )
        ],
        title="Research & Development <2026>",
    )

    assert "Research &amp; Development &lt;2026&gt;" in xml
    assert 'text="tag &amp; &lt;one&gt; &quot;q&quot;"' in xml
    assert 'text="A&amp;B &lt;Plan&gt; &quot;quoted&quot;"' in xml
    assert 'source_id="source&lt;&amp;&gt;&quot;"' in xml
    assert 'source_project="source &amp; project"' in xml

    body = parsed_body(xml)
    tag_outline = body.find("outline")
    assert tag_outline is not None
    unit_outline = tag_outline.find("outline")
    assert unit_outline is not None
    assert unit_outline.attrib == {
        "text": 'A&B <Plan> "quoted"',
        "source_id": 'source<&>"',
        "source_project": "source & project",
        "content_type": "insight",
        "created_at": "2026-05-01T10:15:00+00:00",
        "updated_at": "2026-05-02T08:30:00+00:00",
    }


def test_export_units_to_opml_places_multi_tag_units_under_each_tag():
    xml = export_units_to_opml(
        [
            unit("unit-a", "Shared", ["systems", "ai"]),
            unit("unit-b", "Solo", ["ai"]),
        ]
    )

    body = parsed_body(xml)
    tags = body.findall("outline")

    assert outline_texts(body) == ["ai", "systems"]
    assert outline_texts(tags[0]) == ["Shared", "Solo"]
    assert outline_texts(tags[1]) == ["Shared"]


def test_export_units_to_opml_can_include_or_exclude_untagged_units():
    tagged = unit("unit-a", "Tagged", ["ai"])
    untagged = unit("unit-b", "Untagged Unit", [])

    included = parsed_body(export_units_to_opml([tagged, untagged], include_untagged=True))
    excluded = parsed_body(export_units_to_opml([tagged, untagged], include_untagged=False))

    assert outline_texts(included) == ["ai", "Untagged"]
    assert outline_texts(included.findall("outline")[1]) == ["Untagged Unit"]
    assert outline_texts(excluded) == ["ai"]


def test_export_units_to_opml_orders_tags_and_units_deterministically():
    units = [
        unit("unit-c", "Beta", ["zeta", "alpha"]),
        unit("unit-b", "Alpha", ["zeta"]),
        unit("unit-a", "Alpha", ["alpha"]),
    ]

    first_xml = export_units_to_opml(units)
    second_xml = export_units_to_opml(list(reversed(units)))

    first_body = parsed_body(first_xml)
    tags = first_body.findall("outline")

    assert first_xml == second_xml
    assert outline_texts(first_body) == ["alpha", "zeta"]
    assert outline_texts(tags[0]) == ["Alpha", "Beta"]
    assert outline_texts(tags[1]) == ["Alpha", "Beta"]
