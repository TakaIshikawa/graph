from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO

from graph.export import export_units_to_internal_wikilink_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


@dataclass
class Unit:
    id: str
    content: str
    metadata: dict[str, object]


def test_internal_wikilink_inventory_counts_targets_labels_and_empty_links():
    result = rows(
        export_units_to_internal_wikilink_inventory_csv(
            [
                {"id": "b", "content": "[[Beta]] and [[Alpha|A]] and [[Alpha]] and [[]]"},
                Unit(id="a", content="No links", metadata={"summary": "[[Meta Target|label]]"}),
            ]
        )
    )

    assert result == [
        {"unit_id": "a", "wikilink_count": "1", "unique_targets": "Meta Target", "labeled_link_count": "1", "empty_target_count": "0"},
        {"unit_id": "b", "wikilink_count": "4", "unique_targets": "Alpha; Beta", "labeled_link_count": "1", "empty_target_count": "1"},
    ]


def test_internal_wikilink_inventory_writes_path_metadata(tmp_path):
    output = tmp_path / "links.csv"
    result = export_units_to_internal_wikilink_inventory_csv([{"id": "u1"}], output)

    assert result["path"] == str(output)
    assert result["unit_count"] == 1
    assert result["rows_exported"] == 1
    assert result["bytes_written"] == output.stat().st_size
    assert rows(output.read_text(encoding="utf-8"))[0]["wikilink_count"] == "0"
