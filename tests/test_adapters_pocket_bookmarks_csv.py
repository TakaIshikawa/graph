from graph.adapters import PocketBookmarksCsvAdapter


def test_pocket_bookmarks_csv_ingests_tags_optional_fields_and_stable_ids(tmp_path):
    path = tmp_path / "pocket.csv"
    path.write_text("title,url,tags,time_added,status,excerpt,favorite\nExample,https://e.test,\"Read, Later\",2025-01-02,unread,Summary,yes\nNo Extras,https://n.test,,,,,\n", encoding="utf-8")

    units = PocketBookmarksCsvAdapter(str(path)).ingest().units
    again = PocketBookmarksCsvAdapter(str(path)).ingest().units

    by_title = {unit.title: unit for unit in units}
    assert sorted(by_title) == ["Example", "No Extras"]
    assert by_title["Example"].metadata["url"] == "https://e.test"
    assert by_title["Example"].tags == ["read", "later"]
    assert by_title["Example"].metadata["favorite"] is True
    assert "Summary" in by_title["Example"].content
    assert by_title["No Extras"].tags == []
    assert [unit.source_id for unit in units] == [unit.source_id for unit in again]
