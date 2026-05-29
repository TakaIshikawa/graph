from graph.adapters import RaindropBookmarksCsvAdapter


def test_raindrop_bookmarks_csv_preserves_collection_favorite_and_content(tmp_path):
    path = tmp_path / "raindrop.csv"
    path.write_text("title,link,excerpt,note,tags,collection,created,favorite\nExample,https://e.test,Excerpt,Note,\"A,B\",Research,2025-01-02,true\nBlank,https://b.test,,,,Inbox,,no\n", encoding="utf-8")

    units = RaindropBookmarksCsvAdapter(str(path)).ingest().units

    by_title = {unit.title: unit for unit in units}
    assert by_title["Example"].metadata["collection"] == "Research"
    assert by_title["Example"].metadata["favorite"] is True
    assert by_title["Example"].tags == ["a", "b"]
    assert "Excerpt\nNote" in by_title["Example"].content
    assert "Collection: Research" in by_title["Example"].content
    assert "Note:" not in by_title["Blank"].content
