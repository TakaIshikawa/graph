from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_tag_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_body_hashtags_and_nested_tags():
    result = rows(export_units_to_markdown_tag_csv([{"id": "u", "content": "Body #Topic and #Topic/Sub"}]))
    assert [(row["tag"], row["normalized_tag"], row["depth"]) for row in result] == [("#Topic", "#topic", "1"), ("#Topic/Sub", "#topic/sub", "2")]


def test_excludes_headings_code_fences_inline_code_and_url_fragments():
    content = "# Heading\n`#code`\nhttps://example.test/#frag\n```\n#fence\n```\nBody #real"
    result = rows(export_units_to_markdown_tag_csv([{"id": "u", "content": content}]))
    assert [row["normalized_tag"] for row in result] == ["#real"]
