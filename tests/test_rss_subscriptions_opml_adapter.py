from __future__ import annotations

from graph.adapters.rss_subscriptions_opml import RssSubscriptionsOpmlAdapter


def test_rss_subscriptions_opml_parses_nested_feeds(tmp_path):
    path = tmp_path / "feeds.opml"
    path.write_text(
        """<opml version="2.0"><body>
        <outline text="Tech">
          <outline text="Python">
            <outline text="Py Feed" type="rss" xmlUrl="https://example.com/py.xml" htmlUrl="https://example.com/py"/>
          </outline>
        </outline>
        <outline title="No Site" xmlUrl="https://example.com/feed.xml" category="Solo"/>
        </body></opml>""",
        encoding="utf-8",
    )

    units = RssSubscriptionsOpmlAdapter(path=str(path)).ingest().units

    assert [unit.title for unit in units] == ["Py Feed", "No Site"]
    assert units[0].metadata["xml_url"] == "https://example.com/py.xml"
    assert units[0].metadata["html_url"] == "https://example.com/py"
    assert units[0].metadata["category_path"] == ["Tech", "Python"]
    assert units[0].metadata["outline_type"] == "rss"
    assert units[1].metadata["xml_url"] == "https://example.com/feed.xml"
    assert units[1].metadata["category_path"] == ["Solo"]
    assert "https://example.com/feed.xml" in units[1].content
