from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_hashtag_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_inline_hashtags_with_normalized_values_and_positions():
    result = rows(export_units_to_markdown_hashtag_csv([{"id": "u", "title": "T", "source_project": "docs", "content": "Track #Topic_Tag here\nSecond #tag"}]))

    assert [(row["tag_text"], row["normalized_tag"], row["line_number"], row["column"]) for row in result] == [
        ("#Topic_Tag", "topic-tag", "1", "7"),
        ("#tag", "tag", "2", "8"),
    ]
    assert result[0]["source"] == "docs"


def test_ignores_code_spans_and_url_fragments():
    result = rows(export_units_to_markdown_hashtag_csv([{"id": "u", "content": "`#code` https://example.test/page#frag real #ok"}]))

    assert [(row["tag_text"], row["normalized_tag"]) for row in result] == [("#ok", "ok")]
