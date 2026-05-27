from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO

from graph.export import export_units_to_toc_marker_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


@dataclass
class Unit:
    id: str
    content: str


def test_toc_marker_inventory_counts_marker_families_case_insensitively():
    result = rows(export_units_to_toc_marker_inventory_csv([{"id": "u1", "content": "Intro\n[TOC]\n<!-- toc -->\n[[toc]]"}]))[0]

    assert result == {
        "unit_id": "u1",
        "toc_marker_count": "3",
        "html_toc_comment_count": "1",
        "bracket_toc_marker_count": "2",
        "first_marker_line": "2",
    }


def test_toc_marker_inventory_ignores_fenced_code_and_writes_path(tmp_path):
    output = tmp_path / "toc.csv"

    result = export_units_to_toc_marker_inventory_csv([Unit("o", "```md\n[TOC]\n<!-- toc -->\n```\n[[TOC]]")], output)

    assert result["rows_exported"] == 1
    assert rows(output.read_text(encoding="utf-8"))[0]["toc_marker_count"] == "1"
