from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_yaml_tag_directive_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_yaml_tag_directive_csv_reads_only_opening_frontmatter_block():
    rows = _rows(export_unit_yaml_tag_directive_csv([{"id": "u1", "title": "Yaml", "content": "---\n%TAG !e! tag:example.com,2024:\nname: x\n---\n%TAG !skip! tag:skip:"}]))

    assert rows == [{"unit_id": "u1", "title": "Yaml", "handle": "!e!", "prefix": "tag:example.com,2024:", "line_number": "2"}]
