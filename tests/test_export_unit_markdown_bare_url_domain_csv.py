from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_bare_url_domain_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_bare_urls_with_domains():
    result = rows(export_units_to_markdown_bare_url_domain_csv([{"id": "u", "title": "T", "source": "s", "content": "See https://WWW.Example.COM/a?b=1 and http://sub.test."}]))

    assert [(row["raw_url"], row["scheme"], row["hostname"], row["normalized_domain"], row["has_path_query_fragment"]) for row in result] == [
        ("https://WWW.Example.COM/a?b=1", "https", "www.example.com", "example.com", "True"),
        ("http://sub.test", "http", "sub.test", "sub.test", "False"),
    ]
    assert result[0]["source"] == "s"


def test_ignores_markdown_link_destinations_code_and_fences():
    content = "[x](https://linked.test/p) `https://code.test`\n```md\nhttps://skip.test\n```\nhttps://bare.test/path)"

    result = rows(export_units_to_markdown_bare_url_domain_csv([{"id": "u", "content": content}]))

    assert [(row["raw_url"], row["line_number"]) for row in result] == [("https://bare.test/path", "5")]


def test_path_write_returns_export_metadata(tmp_path):
    output = tmp_path / "url_domains.csv"

    result = export_units_to_markdown_bare_url_domain_csv([{"id": "u", "content": "https://example.test"}], output)

    assert result == {"path": str(output), "unit_count": 1, "rows_exported": 1, "bytes_written": output.stat().st_size}
