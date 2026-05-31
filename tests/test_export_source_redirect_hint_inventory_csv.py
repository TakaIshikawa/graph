from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_sources_to_redirect_hint_inventory_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_redirect_hint_inventory_exports_counts_and_differing_urls():
    text = export_sources_to_redirect_hint_inventory_csv(
        [
            {"source_id": "b", "name": "B", "metadata": {"original_url": "https://old.test", "final_url": "https://new.test"}},
            {"source_id": "a", "metadata": {"url": "https://same.test", "redirected_url": "https://same.test", "redirect_count": 2, "status_code": 301}},
            {"source_id": "plain", "metadata": {"url": "https://plain.test"}},
        ]
    )

    assert _rows(text) == [
        {
            "source_id": "a",
            "name": "",
            "original_url": "https://same.test",
            "final_url": "https://same.test",
            "redirect_count": "2",
            "status_code": "301",
        },
        {
            "source_id": "b",
            "name": "B",
            "original_url": "https://old.test",
            "final_url": "https://new.test",
            "redirect_count": "",
            "status_code": "",
        },
    ]


def test_redirect_hint_inventory_top_level_precedence_and_path_stats(tmp_path):
    path = tmp_path / "redirects.csv"
    sources = [
        {
            "source_id": "s",
            "original_url": "https://top-old.test",
            "final_url": "https://top-new.test",
            "redirect_count": 1,
            "status_code": 302,
            "metadata": {"original_url": "https://meta-old.test", "final_url": "https://meta-new.test"},
        }
    ]

    expected = export_sources_to_redirect_hint_inventory_csv(sources)
    stats = export_sources_to_redirect_hint_inventory_csv(sources, path)

    assert _rows(expected)[0]["original_url"] == "https://top-old.test"
    assert _rows(expected)[0]["final_url"] == "https://top-new.test"
    assert path.read_text(encoding="utf-8") == expected
    assert stats["source_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
