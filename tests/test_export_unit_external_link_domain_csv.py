from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_external_link_domains_to_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_external_link_domain_csv_includes_http_markdown_and_autolinks_with_counts():
    text = export_unit_external_link_domains_to_csv(
        [
            {
                "id": "u",
                "title": "Unit",
                "content": "[A](https://Example.COM:443/a) [B](http://sub.example.com/b \"B\") [rel](/x) <https://news.example.co.uk/p>\n```\n[skip](https://skip.test)\n```",
            }
        ]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "Unit", "url": "https://Example.COM:443/a", "domain": "example.com", "registrable_host": "example.com", "line_number": "1", "link_text": "A", "domain_count_within_unit": "1"},
        {"unit_id": "u", "title": "Unit", "url": "https://news.example.co.uk/p", "domain": "news.example.co.uk", "registrable_host": "example.co.uk", "line_number": "1", "link_text": "https://news.example.co.uk/p", "domain_count_within_unit": "1"},
        {"unit_id": "u", "title": "Unit", "url": "http://sub.example.com/b", "domain": "sub.example.com", "registrable_host": "example.com", "line_number": "1", "link_text": "B", "domain_count_within_unit": "1"},
    ]


def test_external_link_domain_csv_counts_duplicate_domains_within_unit():
    rows = _rows(export_unit_external_link_domains_to_csv([{"id": "u", "content": "[A](https://example.com/a) [B](https://EXAMPLE.com/b)"}]))

    assert [row["domain_count_within_unit"] for row in rows] == ["2", "2"]
