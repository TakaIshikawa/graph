from __future__ import annotations

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.splitwise_expenses_csv import SplitwiseExpensesCsvAdapter
from graph.types.enums import SourceProject


def test_splitwise_expenses_csv_ingests_expense_rows(tmp_path):
    export = tmp_path / "splitwise.csv"
    export.write_text(
        "Date,Expense ID,Description,Category,Cost,Currency,Group,Paid By,Owed By,Users,Comments,Settled\n"
        "2026-05-01,EXP1,Dinner,Food,72.30,USD,Trip,Ada,\"Grace, Linus\",\"Ada;Grace;Linus\",Shared meal,false\n",
        encoding="utf-8",
    )

    result = SplitwiseExpensesCsvAdapter(path=str(export)).ingest(entity_types=["expense"])

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.SPLITWISE_EXPENSES_CSV
    assert unit.source_id == "splitwise_expenses_csv:EXP1"
    assert unit.metadata["timestamp"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["description"] == "Dinner"
    assert unit.metadata["category"] == "Food"
    assert unit.metadata["cost"] == 72.3
    assert unit.metadata["currency"] == "USD"
    assert unit.metadata["group"] == "Trip"
    assert unit.metadata["paid_by"] == "Ada"
    assert unit.metadata["owed_by"] == "Grace, Linus"
    assert unit.metadata["users"] == ["Ada", "Grace", "Linus"]
    assert "Grace" in unit.content
    assert "Trip" in unit.content
    assert "splitwise" in unit.tags
    assert "expense" in unit.tags


def test_splitwise_expenses_csv_filters_entity_types(tmp_path):
    export = tmp_path / "splitwise.csv"
    export.write_text("Date,Description,Cost\n2026-05-01,Coffee,5\n", encoding="utf-8")

    assert SplitwiseExpensesCsvAdapter(path=str(export)).ingest(entity_types=["transaction"]).units == []


def test_splitwise_expenses_csv_is_registered():
    assert "splitwise_expenses_csv" in list_adapters()
    assert isinstance(get_adapter("splitwise-expenses-csv"), SplitwiseExpensesCsvAdapter)
