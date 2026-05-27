from graph.adapters import PinboardBookmarksCsvAdapter


def test_pinboard_bookmarks_csv_splits_tags_and_booleans(tmp_path):
    path = tmp_path / "pinboard.csv"
    path.write_text("href,description,extended,tags,time,shared,toread,private\nhttps://e.test,Example,Notes,one two,2025-01-02,yes,0,true\nhttps://n.test,No Notes,,solo,,false,true,false\n", encoding="utf-8")

    units = PinboardBookmarksCsvAdapter(str(path)).ingest().units

    by_title = {unit.title: unit for unit in units}
    assert by_title["Example"].tags == ["one", "two"]
    assert by_title["Example"].metadata["shared"] is True
    assert by_title["Example"].metadata["toread"] is False
    assert by_title["Example"].metadata["private"] is True
    assert "Notes" in by_title["Example"].content
    assert by_title["No Notes"].content == "No Notes\nURL: https://n.test\nTags: solo"
