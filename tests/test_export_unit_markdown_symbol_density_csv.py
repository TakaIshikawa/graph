from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO

from graph.export import export_units_to_markdown_symbol_density_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


@dataclass
class Unit:
    id: str
    content: str


def test_symbol_density_counts_structural_symbols_outside_fences():
    result = rows(export_units_to_markdown_symbol_density_csv([{"id": "u1", "content": "# A [x](y)\n```md\n### ~~ignored~~\n```\n> ==ok=="}]))[0]

    assert result["unit_id"] == "u1"
    assert result["markdown_symbol_count"] == "10"
    assert result["dominant_symbol"] == "="


def test_symbol_density_handles_empty_content_and_path_write(tmp_path):
    output = tmp_path / "symbols.csv"

    result = export_units_to_markdown_symbol_density_csv([Unit("o", "")], output)

    assert result["path"] == str(output)
    assert rows(output.read_text(encoding="utf-8"))[0] == {
        "unit_id": "o",
        "content_length": "0",
        "markdown_symbol_count": "0",
        "symbol_density_per_1k_chars": "0.00",
        "dominant_symbol": "",
    }
