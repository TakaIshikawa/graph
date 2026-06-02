from __future__ import annotations

from graph.adapters.browser_history_csv import BrowserHistoryCsvAdapter


def test_browser_history_csv_parses_visited_at_and_last_visit_time(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text(
        "url,title,visit_count,visited_at,last_visit_time\n"
        "https://example.com/a,Page A,3,2026-01-01T12:00:00Z,\n"
        "https://example.com/b,Page B,7,,2026-01-02T12:00:00Z\n",
        encoding="utf-8",
    )

    units = BrowserHistoryCsvAdapter(path=str(path)).ingest().units

    assert [unit.title for unit in units] == ["Page A", "Page B"]
    assert units[0].metadata["url"] == "https://example.com/a"
    assert units[0].metadata["visit_count"] == "3"
    assert units[0].metadata["visited_at"] == "2026-01-01T12:00:00Z"
    assert units[1].metadata["visit_count"] == "7"
    assert units[1].metadata["visited_at"] == "2026-01-02T12:00:00Z"
    assert units[1].metadata["last_visit_time"] == "2026-01-02T12:00:00Z"


def test_browser_history_csv_blank_titles_use_url_fallback_and_stable_ids(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text(
        "url,title,visit_count,visited_at\n"
        "https://example.com/docs/page,,not-a-number,2026-01-01T12:00:00Z\n"
        "example.org/path?q=1,,5,2026-01-01T13:00:00Z\n",
        encoding="utf-8",
    )

    first, second = BrowserHistoryCsvAdapter(path=str(path)).ingest().units

    assert first.title == "example.com/docs/page"
    assert second.title == "example.org/path"
    assert first.metadata["visit_count"] == "not-a-number"
    assert first.source_id == BrowserHistoryCsvAdapter(path=str(path)).ingest().units[0].source_id
    assert first.source_id != second.source_id
    assert "URL: https://example.com/docs/page" in first.content


def test_browser_history_csv_missing_optional_fields_do_not_fail(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("url,title\nhttps://example.com,Home\n", encoding="utf-8")

    unit = BrowserHistoryCsvAdapter(path=str(path)).ingest().units[0]

    assert unit.title == "Home"
    assert unit.metadata["url"] == "https://example.com"
    assert "visit_count" not in unit.metadata
