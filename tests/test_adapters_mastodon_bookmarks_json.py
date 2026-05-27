import json

from graph.adapters import MastodonBookmarksJsonAdapter


def test_mastodon_bookmarks_json_strips_html_and_preserves_metadata(tmp_path):
    path = tmp_path / "bookmarks.json"
    path.write_text(json.dumps({"bookmarks": [{"id": "1", "url": "https://m.test/1", "account": {"display_name": "Ada", "acct": "ada@example"}, "created_at": "2025-01-02T00:00:00Z", "content": "<p>Hello <strong>world</strong></p>", "tags": [{"name": "ai"}], "language": "en", "reblogs_count": 2, "favourites_count": 3, "replies_count": 4}, {"id": "2", "content": ""}]}), encoding="utf-8")

    unit = MastodonBookmarksJsonAdapter(str(path)).ingest().units[0]

    assert unit.title.startswith("Ada: Hello world")
    assert unit.content == "Author: Ada\nHello world\nURL: https://m.test/1\nTags: ai"
    assert "<p>" not in unit.content
    assert unit.metadata["favourites_count"] == 3
    assert unit.tags == ["ai"]
