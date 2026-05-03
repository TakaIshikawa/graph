from __future__ import annotations

from pathlib import Path

import pytest

from graph.adapters.opml import OpmlAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject


def write_opml(path: Path, body: str) -> Path:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
        <opml version="2.0">
          <body>
            {body}
          </body>
        </opml>
        """,
        encoding="utf-8",
    )
    return path


def test_nested_outlines_preserve_hierarchy_and_url_metadata(tmp_path):
    opml_path = write_opml(
        tmp_path / "reading-list.opml",
        """
        <outline text="Research">
          <outline title="AI">
            <outline text="AI Feed" type="rss"
              xmlUrl="https://example.com/ai.xml"
              htmlUrl="https://example.com/ai"
              url="https://example.com/ai-page" />
          </outline>
        </outline>
        """,
    )

    result = OpmlAdapter(path=str(opml_path)).ingest()

    assert [unit.title for unit in result.units] == ["Research", "AI", "AI Feed"]
    feed = result.units[2]
    assert feed.source_project == SourceProject.OPML
    assert feed.source_entity_type == "outline"
    assert feed.metadata["path"] == "Research/AI/AI Feed"
    assert feed.metadata["path_parts"] == ["Research", "AI", "AI Feed"]
    assert feed.metadata["url"] == "https://example.com/ai-page"
    assert feed.metadata["xmlUrl"] == "https://example.com/ai.xml"
    assert feed.metadata["htmlUrl"] == "https://example.com/ai"
    assert feed.tags == ["Research", "Research/AI", "Research/AI/AI Feed"]
    assert "https://example.com/ai.xml" in feed.content

    assert [(edge.relation, edge.source) for edge in result.edges] == [
        (EdgeRelation.CONTAINS, EdgeSource.SOURCE),
        (EdgeRelation.CONTAINS, EdgeSource.SOURCE),
    ]


def test_malformed_opml_returns_empty_result_with_warning(tmp_path):
    opml_path = tmp_path / "broken.opml"
    opml_path.write_text("<opml><body><outline text='broken'></body>", encoding="utf-8")

    with pytest.warns(UserWarning, match="Skipping invalid OPML file"):
        result = OpmlAdapter(path=str(opml_path)).ingest()

    assert result.units == []
    assert result.edges == []


def test_opml_adapter_is_registered():
    assert "opml" in list_adapters()
    assert isinstance(get_adapter("opml", path="/tmp/feeds.opml"), OpmlAdapter)
