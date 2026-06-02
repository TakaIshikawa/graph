from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_link_utm_parameters_to_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_link_utm_parameter_csv_exports_deduped_sorted_parameter_names():
    text = export_unit_markdown_link_utm_parameters_to_csv(
        [
            {"id": "u", "title": "Unit", "content": "[A](https://e.test/?utm_source=x&utm_medium=y&utm_source=z) [B](https://e.test/?x=1)\n```\n[C](https://e.test/?utm_campaign=x)\n```"},
        ]
    )

    assert rows(text) == [
        {"unit_id": "u", "title": "Unit", "line_number": "1", "link_text": "A", "url": "https://e.test/?utm_source=x&utm_medium=y&utm_source=z", "parameter_names": "utm_medium; utm_source"},
    ]
