from __future__ import annotations

from graph.adapters.microsoft_edge_bookmarks_json import MicrosoftEdgeBookmarksJsonAdapter


def test_microsoft_edge_bookmarks_json_ingests_nested_urls(tmp_path):
    export = tmp_path / "bookmarks.json"
    export.write_text('{"roots":{"bookmark_bar":{"name":"Favorites bar","children":[{"name":"Folder","children":[{"type":"url","name":"Example","url":"https://example.com","guid":"g1","date_added":"13222310400000000"}]}]}}}', encoding="utf-8")

    units = MicrosoftEdgeBookmarksJsonAdapter(path=str(export)).ingest().units

    assert len(units) == 1
    assert units[0].title == "Example"
    assert units[0].metadata["folder_path"] == "Favorites bar/Folder"
    assert units[0].metadata["guid"] == "g1"


def test_microsoft_edge_bookmarks_json_ignores_folders_and_malformed(tmp_path):
    folders = tmp_path / "folders.json"
    folders.write_text('{"roots":{"bookmark_bar":{"name":"Favorites bar","children":[{"name":"Folder","children":[]}]}}}', encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")

    assert MicrosoftEdgeBookmarksJsonAdapter(path=str(folders)).ingest().units == []
    assert MicrosoftEdgeBookmarksJsonAdapter(path=str(bad)).ingest().units == []
