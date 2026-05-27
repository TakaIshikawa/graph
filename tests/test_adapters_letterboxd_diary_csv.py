from graph.adapters import LetterboxdDiaryCsvAdapter


def test_letterboxd_diary_csv_normalizes_metadata_and_review_content(tmp_path):
    path = tmp_path / "diary.csv"
    path.write_text("Watched Date,Name,Year,Rating,Rewatch,Tags,Review\n2025-01-02,Movie,1999,4.5,Yes,\"noir,classic\",Great.\n2025-01-03,Silent,2001,,,,\n", encoding="utf-8")

    units = LetterboxdDiaryCsvAdapter(str(path)).ingest().units

    assert units[0].title == "Movie (1999)"
    assert units[0].metadata["rating"] == 4.5
    assert units[0].metadata["rewatch"] is True
    assert units[0].tags == ["noir", "classic"]
    assert "Great." in units[0].content
    assert units[1].title == "Silent (2001)"
