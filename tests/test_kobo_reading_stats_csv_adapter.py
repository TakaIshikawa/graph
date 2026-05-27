from __future__ import annotations

from graph.adapters.kobo_reading_stats_csv import KoboReadingStatsCsvAdapter
from graph.adapters.registry import get_adapter


def test_kobo_reading_stats_csv_normalizes_progress_and_stable_ids_without_isbn(tmp_path):
    path = tmp_path / "stats.csv"
    path.write_text("Book Title,Author,Percent Read,Minutes Read,Last Read Date,Status,Shelves\nBook,Ada,42%,90,2026-01-02,Reading,Fiction;Favorites\n", encoding="utf-8")

    unit = KoboReadingStatsCsvAdapter(str(path)).ingest().units[0]

    assert unit.source_id.startswith("kobo_reading_stats_csv:")
    assert unit.metadata["percent_read"] == 42.0
    assert unit.metadata["last_read_at"] == "2026-01-02T00:00:00+00:00"
    assert unit.metadata["shelves"] == ["Fiction", "Favorites"]
    assert isinstance(get_adapter("kobo_reading_stats_csv"), KoboReadingStatsCsvAdapter)
