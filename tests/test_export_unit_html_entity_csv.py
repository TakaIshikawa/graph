from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_html_entity_csv import export_units_to_html_entity_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_html_entity_csv_classifies_entities_and_skips_fenced_code():
    text = export_units_to_html_entity_csv(
        [
            {"id": "u1", "title": "One", "content": "Use &amp; and &#169;.\n```\nSkip &lt;\n```\nHex &#x1F600;"},
        ]
    )

    assert rows(text) == [
        {"unit_id": "u1", "title": "One", "line_number": "1", "entity": "&#169;", "entity_type": "decimal", "decoded": "©"},
        {"unit_id": "u1", "title": "One", "line_number": "1", "entity": "&amp;", "entity_type": "named", "decoded": "&"},
        {"unit_id": "u1", "title": "One", "line_number": "5", "entity": "&#x1F600;", "entity_type": "hex", "decoded": "😀"},
    ]
