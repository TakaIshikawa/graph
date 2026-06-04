from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_iframe_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_iframe_csv_exports_attributes_and_domain():
    text = export_units_to_markdown_html_iframe_csv(
        [
            {
                "id": "u",
                "metadata": {"title": "Embed", "path": "unit.md", "source_name": "web"},
                "content": '<iframe src="https://Example.com/watch?id=1" title="Player" loading="lazy" sandbox allow="fullscreen" '
                'referrerpolicy="no-referrer" width="560" height="315"></iframe>',
            }
        ]
    )

    assert _rows(text) == [
        {
            "unit_id": "u",
            "title": "Embed",
            "source_path": "unit.md",
            "source": "web",
            "line_number": "1",
            "src": "https://Example.com/watch?id=1",
            "title": "Player",
            "loading": "lazy",
            "sandbox": "",
            "allow": "fullscreen",
            "referrerpolicy": "no-referrer",
            "width": "560",
            "height": "315",
            "domain": "example.com",
        }
    ]


def test_iframe_csv_ignores_fences_and_allows_missing_attributes():
    text = export_units_to_markdown_html_iframe_csv([{"id": "u", "content": "```\n<iframe src='skip'></iframe>\n```\n<iframe></iframe>"}])

    assert _rows(text) == [
        {
            "unit_id": "u",
            "title": "",
            "source_path": "",
            "source": "",
            "line_number": "4",
            "src": "",
            "title": "",
            "loading": "",
            "sandbox": "",
            "allow": "",
            "referrerpolicy": "",
            "width": "",
            "height": "",
            "domain": "",
        }
    ]
