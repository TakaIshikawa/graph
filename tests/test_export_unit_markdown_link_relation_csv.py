from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_link_relation_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_markdown_link_relation_csv_splits_rel_tokens_and_handles_quotes_case():
    rows = _rows(export_unit_markdown_link_relation_csv([{"id": "u1", "title": "Links", "content": "<a HREF='https://a.test' REL=\"nofollow noopener\">A</a>\n<a href=\"/b\" rel='tag'>B</a>\n<a href=\"/c\">C</a><a rel=\"next\">D</a>"}]))

    assert rows == [
        {"unit_id": "u1", "title": "Links", "href": "https://a.test", "rel_value": "nofollow noopener", "rel_token": "nofollow", "line_number": "1"},
        {"unit_id": "u1", "title": "Links", "href": "https://a.test", "rel_value": "nofollow noopener", "rel_token": "noopener", "line_number": "1"},
        {"unit_id": "u1", "title": "Links", "href": "/b", "rel_value": "tag", "rel_token": "tag", "line_number": "2"},
    ]
