from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from graph.export import export_units_to_rss
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    title: str | None = None,
    content: str = "Content body.",
    tags: list[str] | None = None,
    metadata: dict | None = None,
    created_at: datetime = UNIT_TIME,
    updated_at: datetime = UNIT_TIME,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.FEED,
        source_id=f"source-{unit_id}",
        source_entity_type="entry",
        title=title or f"Title {unit_id}",
        content=content,
        content_type=ContentType.ARTIFACT,
        tags=tags or [],
        created_at=created_at,
        ingested_at=created_at,
        updated_at=updated_at,
        metadata=metadata or {},
    )


def test_export_units_to_rss_creates_valid_rss_feed():
    root = ET.fromstring(export_units_to_rss(unit("a")))

    assert root.tag == "rss"
    assert root.get("version") == "2.0"
    assert root.find("channel/title").text == "Knowledge Graph"
    assert root.find("channel/lastBuildDate").text == "Fri, 01 May 2026 10:15:00 GMT"


def test_export_units_to_rss_preserves_input_order_and_item_fields():
    xml = export_units_to_rss(
        [
            unit("b", title="Second"),
            unit(
                "a",
                title="First",
                content="Alpha < beta",
                tags=["rss", "feeds"],
                metadata={"url": "https://example.test/a"},
            ),
        ],
        channel_title="My Feed",
        channel_link="https://example.test/",
        channel_description="Portable units",
    )

    channel = ET.fromstring(xml).find("channel")
    assert channel.find("title").text == "My Feed"
    assert channel.find("link").text == "https://example.test/"
    assert channel.find("description").text == "Portable units"

    items = channel.findall("item")
    assert [item.find("title").text for item in items] == ["Second", "First"]
    assert items[1].find("link").text == "https://example.test/a"
    assert items[1].find("guid").text == "https://example.test/a"
    assert items[1].find("pubDate").text == "Fri, 01 May 2026 10:15:00 GMT"
    assert [cat.text for cat in items[1].findall("category")] == ["rss", "feeds"]
    assert items[1].find("description").text == "Alpha < beta"


def test_export_units_to_rss_uses_non_permalink_guid_without_url():
    item = ET.fromstring(export_units_to_rss(unit("a"))).find("channel/item")

    assert item.find("guid").text == "source-a"
    assert item.find("guid").get("isPermaLink") == "false"
    assert item.find("link") is None


def test_export_units_to_rss_writes_to_file(tmp_path):
    path = tmp_path / "feed.xml"

    xml = export_units_to_rss([unit("a")], path)

    assert path.read_text(encoding="utf-8") == xml


def test_export_units_to_rss_accepts_single_unit():
    root = ET.fromstring(export_units_to_rss(unit("a")))

    assert len(root.findall("channel/item")) == 1
