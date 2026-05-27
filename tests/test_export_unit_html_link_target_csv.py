from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_html_link_target_csv import export_units_to_html_link_target_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_units_to_html_link_target_csv_header_only_for_empty_input():
    assert export_units_to_html_link_target_csv([]) == "unit_id,title,target,target_type,anchor_text,source_field\n"


def test_export_units_to_html_link_target_csv_extracts_content_and_metadata_links_sorted():
    text = export_units_to_html_link_target_csv(
        [
            {
                "id": "u1",
                "title": "Links",
                "content": '<a href="/local">Local <b>page</b></a><a href="mailto:a@example.test">Mail</a>',
                "metadata": {"html": '<a href="https://example.test/path">External</a>', "body": '<a href="#top">Top</a>'},
            }
        ]
    )

    assert _rows(text) == [
        {"unit_id": "u1", "title": "Links", "target": "#top", "target_type": "fragment", "anchor_text": "Top", "source_field": "metadata.body"},
        {"unit_id": "u1", "title": "Links", "target": "/local", "target_type": "internal", "anchor_text": "Local page", "source_field": "content"},
        {"unit_id": "u1", "title": "Links", "target": "https://example.test/path", "target_type": "external", "anchor_text": "External", "source_field": "metadata.html"},
        {"unit_id": "u1", "title": "Links", "target": "mailto:a@example.test", "target_type": "mailto", "anchor_text": "Mail", "source_field": "content"},
    ]
