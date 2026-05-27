from __future__ import annotations

from datetime import datetime, timedelta, timezone

from graph.adapters.hacker_news_saved_html import HackerNewsSavedHtmlAdapter
from graph.types.models import SyncState


def test_hacker_news_saved_html_ingests_story_metadata(tmp_path):
    page = tmp_path / "saved.html"
    page.write_text(
        """
<table>
  <tr class="athing" id="12345">
    <td class="title"><span class="titleline"><a href="https://example.com/post">Example Story</a></span></td>
  </tr>
  <tr><td class="subtext">
    <span class="score">42 points</span>
    <a href="user?id=taka">taka</a>
    <span class="age" title="2025-01-02T03:04:05"><a href="item?id=12345">1 day ago</a></span>
    <a href="item?id=12345">7 comments</a>
  </td></tr>
</table>
""",
        encoding="utf-8",
    )

    unit = HackerNewsSavedHtmlAdapter(path=str(page)).ingest().units[0]

    assert unit.source_project == "hacker_news_saved_html"
    assert unit.source_id == "hacker_news_saved_html:12345"
    assert unit.source_entity_type == "saved_story"
    assert unit.title == "Example Story"
    assert unit.metadata["url"] == "https://example.com/post"
    assert unit.metadata["hn_item_id"] == 12345
    assert unit.metadata["points"] == 42
    assert unit.metadata["author"] == "taka"
    assert unit.metadata["comments_count"] == 7


def test_hacker_news_saved_html_tolerates_missing_optional_fields_and_filters(tmp_path):
    page = tmp_path / "saved.html"
    page.write_text(
        """
<tr class="athing"><td><span class="titleline"><a href="https://example.com/a">A</a></span></td></tr>
<tr class="athing" id="222"><td><span class="titleline"><a href="item?id=222">Ask HN</a></span></td></tr>
""",
        encoding="utf-8",
    )

    units = HackerNewsSavedHtmlAdapter(path=str(page)).ingest().units

    assert len(units) == 2
    assert units[0].source_id == HackerNewsSavedHtmlAdapter()._source_id("", "https://example.com/a")
    assert units[1].source_id == "hacker_news_saved_html:222"
    assert HackerNewsSavedHtmlAdapter(path=str(page)).ingest(entity_types=["bookmark"]).units == []

    since = SyncState(
        source_project="hacker_news_saved_html",
        source_entity_type="saved_story",
        last_sync_at=datetime.now(timezone.utc) + timedelta(days=1),
    )
    assert HackerNewsSavedHtmlAdapter(path=str(page)).ingest(since=since).units == []
