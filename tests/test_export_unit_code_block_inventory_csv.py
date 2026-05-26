from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO

from graph.export import export_units_to_code_block_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


@dataclass
class Unit:
    id: str
    content: str | None = None


def test_code_block_inventory_counts_backtick_tilde_unlabeled_and_unterminated():
    result = rows(export_units_to_code_block_inventory_csv([{"id": "u", "content": "```Python\nx\n```\n~~~\na\nb\n~~~\n```js\nz"}]))[0]

    assert result == {"unit_id": "u", "code_block_count": "3", "languages": "js; python", "unlabeled_block_count": "1", "max_block_line_count": "2"}


def test_code_block_inventory_supports_object_and_path_write(tmp_path):
    output = tmp_path / "code.csv"
    result = export_units_to_code_block_inventory_csv([Unit("o")], output)

    assert result["unit_count"] == 1
    assert result["rows_exported"] == 1
    assert rows(output.read_text(encoding="utf-8"))[0]["code_block_count"] == "0"
