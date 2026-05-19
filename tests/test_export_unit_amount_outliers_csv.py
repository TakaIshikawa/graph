from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_amount_outliers_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, amount: object, *, category: str = "Food") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(id=unit_id, source_project="Bank", source_id=f"source-{unit_id}", source_entity_type="transaction", title=f"Title {unit_id}", content="", metadata={"amount": amount, "account": "Card", "category": category, "currency": "USD"}, tags=[])


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_amount_outliers_uses_median_multiple_defaults():
    text = export_unit_amount_outliers_csv([unit("a", "10"), unit("b", "$12.00"), unit("c", "($100.00)")])

    assert rows(text) == [
        {
            "unit_id": "c",
            "source_project": "Bank",
            "account": "Card",
            "category": "Food",
            "currency": "USD",
            "amount": "-100",
            "group_median_amount": "12",
            "multiple_of_median": "8.333333333333333333333333333",
            "threshold": "3",
            "title": "Title c",
        }
    ]


def test_amount_outliers_minimum_group_size_threshold_and_stats(tmp_path):
    units = [unit("a", "10"), unit("b", "50")]
    expected = export_unit_amount_outliers_csv(units, minimum_group_size=2, median_multiple_threshold=2)
    path = tmp_path / "outliers.csv"

    stats = export_unit_amount_outliers_csv(units, path, minimum_group_size=2, median_multiple_threshold=2)

    assert rows(expected) == []
    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 2
    assert stats["rows_exported"] == 0
    assert stats["bytes_written"] == path.stat().st_size
