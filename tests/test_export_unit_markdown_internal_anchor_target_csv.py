from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_internal_anchor_target_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_internal_anchor_target_csv_exports_targets_and_duplicates_outside_fences():
    text = export_units_to_markdown_internal_anchor_target_csv(
        [{"id": "u", "title": "Doc", "content": "# Main Heading\n## Other {#main-heading}\nParagraph ^block1\n```md\n# Ignored\n^ignored\n```"}]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "Doc", "line_number": "1", "target_type": "heading", "target": "main-heading", "label": "Main Heading", "duplicate_in_unit": "true"},
        {"unit_id": "u", "title": "Doc", "line_number": "2", "target_type": "custom_id", "target": "main-heading", "label": "Other", "duplicate_in_unit": "true"},
        {"unit_id": "u", "title": "Doc", "line_number": "2", "target_type": "heading", "target": "other", "label": "Other", "duplicate_in_unit": "false"},
        {"unit_id": "u", "title": "Doc", "line_number": "3", "target_type": "block_id", "target": "block1", "label": "Paragraph", "duplicate_in_unit": "false"},
    ]


def test_internal_anchor_target_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "anchors.csv"
    units = [{"id": "u", "content": "# One"}]

    expected = export_units_to_markdown_internal_anchor_target_csv(units)
    stats = export_units_to_markdown_internal_anchor_target_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
