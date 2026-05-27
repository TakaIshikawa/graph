from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_markdown_emphasis_inventory_csv import export_units_to_markdown_emphasis_inventory_csv


def test_export_units_to_markdown_emphasis_inventory_csv_counts_markers_excluding_code():
    rows = list(
        csv.DictReader(
            StringIO(
                export_units_to_markdown_emphasis_inventory_csv(
                    [{"id": "u1", "content": "**bold** and *it* and ~~gone~~ and ==mark== and `**no**`\n```\n*no*\n```"}]
                )
            )
        )
    )

    assert rows == [{"unit_id": "u1", "bold_count": "1", "italic_count": "1", "strikethrough_count": "1", "highlight_count": "1"}]
