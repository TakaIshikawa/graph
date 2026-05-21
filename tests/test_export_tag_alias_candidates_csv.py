from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_tag_alias_candidates_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_tag_alias_candidates_csv_groups_separator_and_case_variants():
    text = export_tag_alias_candidates_csv(
        [
            {"id": "1", "source_project": "A", "tags": ["Machine Learning"]},
            {"id": "2", "source_project": "B", "tags": ["machine-learning"]},
            {"id": "3", "source_project": "B", "metadata": {"tag": "machine_learning"}},
        ]
    )

    result = rows(text)
    assert [row["variant_tag"] for row in result] == ["machine-learning", "machine_learning"]
    assert result[0]["canonical_tag"] == "Machine Learning"
    assert result[0]["reason"] == "separator"


def test_export_tag_alias_candidates_csv_empty_input_returns_header():
    assert export_tag_alias_candidates_csv([]) == "canonical_tag,variant_tag,unit_count,source_count,reason\n"

