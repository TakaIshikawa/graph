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


def test_splitwise_expenses_csv_emits_group_aggregates_and_edges(tmp_path):
    export = tmp_path / "splitwise.csv"
    export.write_text(
        "Date,Expense ID,Description,Category,Cost,Currency,Group,Paid By,Owed By,Users\n"
        "2026-05-01,EXP1,Dinner,Food,72.30,USD,Trip,Ada,\"Grace, Linus\",Ada;Grace;Linus\n"
        "2026-05-03,EXP2,Taxi,Travel,30.00,USD, trip ,Grace,Ada,Ada;Grace\n"
        "2026-05-04,EXP3,Coffee,Food,5.00,USD,,Ada,Grace,Ada;Grace\n",
        encoding="utf-8",
    )

    result = SplitwiseExpensesCsvAdapter(path=str(export)).ingest(entity_types=["expense", "group"])

    groups = [unit for unit in result.units if unit.source_entity_type == "group"]
    assert len(groups) == 1
    group = groups[0]
    assert group.metadata["group"] == "Trip"
    assert group.metadata["expense_count"] == 2
    assert group.metadata["participants"] == ["Ada", "Grace", "Grace, Linus", "Linus"]
    assert group.metadata["categories"] == ["Food", "Travel"]
    assert group.metadata["total_cost"] == 102.3
    assert group.metadata["currencies"] == ["USD"]
    assert group.metadata["first_seen"] == "2026-05-01T00:00:00+00:00"
    assert group.metadata["last_seen"] == "2026-05-03T00:00:00+00:00"
    assert {(edge.from_unit_id, edge.to_unit_id, edge.metadata["relation_type"]) for edge in result.edges} == {
        ("splitwise_expenses_csv:EXP1", group.source_id, "expense_group"),
        ("splitwise_expenses_csv:EXP2", group.source_id, "expense_group"),
    }

    assert [unit.source_entity_type for unit in SplitwiseExpensesCsvAdapter(path=str(export)).ingest(entity_types=["group"]).units] == ["group"]
    assert SplitwiseExpensesCsvAdapter(path=str(export)).ingest(entity_types=["expense"]).edges == []


def test_splitwise_expenses_csv_is_registered():
    assert "splitwise_expenses_csv" in list_adapters()
    assert isinstance(get_adapter("splitwise-expenses-csv"), SplitwiseExpensesCsvAdapter)
