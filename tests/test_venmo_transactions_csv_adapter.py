from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.venmo_transactions_csv import VenmoTransactionsCsvAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_venmo_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "venmo.csv"
    export.write_text(
        "Datetime,Transaction ID,Type,Status,Note,From,To,Amount,Fee,Funding Source,Destination,Privacy\n"
        "2026-05-01T08:00:00Z,V1,Payment,Complete,Lunch,Ada,Grace,$12.50,$0.25,Bank,Venmo balance,Friends\n",
        encoding="utf-8",
    )

    result = VenmoTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.VENMO_TRANSACTIONS_CSV
    assert unit.source_id == "venmo_transactions_csv:V1"
    assert unit.metadata["timestamp"] == "2026-05-01T08:00:00+00:00"
    assert unit.metadata["type"] == "Payment"
    assert unit.metadata["status"] == "Complete"
    assert unit.metadata["note"] == "Lunch"
    assert unit.metadata["from"] == "Ada"
    assert unit.metadata["to"] == "Grace"
    assert unit.metadata["amount"] == 12.5
    assert unit.metadata["fee"] == 0.25
    assert unit.metadata["funding_source"] == "Bank"
    assert unit.metadata["destination"] == "Venmo balance"
    assert unit.metadata["counterparty"] == "Ada"
    assert unit.metadata["privacy"] == "Friends"


def test_venmo_transactions_csv_preserves_currency_url_and_source_file(tmp_path):
    export = tmp_path / "venmo.csv"
    export.write_text(
        "Date,Transaction ID,Type,Note,Amount,Currency,URL\n"
        "2026-05-01,V1,Payment,Lunch,12.50,USD,https://venmo.example/tx/V1\n",
        encoding="utf-8-sig",
    )

    unit = VenmoTransactionsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.metadata["currency"] == "USD"
    assert unit.metadata["url"] == "https://venmo.example/tx/V1"
    assert unit.metadata["source_file"] == "venmo.csv"


def test_venmo_transactions_csv_uses_digest_fallback_and_negative_amounts(tmp_path):
    export = tmp_path / "venmo.csv"
    export.write_text(
        "Date,Type,Note,From,To,Amount,Fee\n"
        "2026-05-02,Charge,Rent,Grace,Ada,($500.00),\n",
        encoding="utf-8",
    )

    result = VenmoTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_id.startswith("venmo_transactions_csv:")
    assert unit.source_id != "venmo_transactions_csv:"
    assert unit.metadata["amount"] == -500.0
    assert "fee" not in unit.metadata


def test_venmo_transactions_csv_directory_bom_invalid_rows_stable_ids_counterparty_and_since(tmp_path):
    first_export = tmp_path / "first.csv"
    second_export = tmp_path / "second.csv"
    first_export.write_text(
        "\ufeffDate,ID,Type,Status,Counterparty,Note,Amount,Currency,URL\n"
        "2026-05-01,V1,Payment,Complete,Grace,Lunch,$12.50,USD,https://venmo.example/tx/V1\n"
        "2026-05-02,V2,Payment,Complete,Linus,Coffee,,USD,\n",
        encoding="utf-8",
    )
    second_export.write_text(
        "Datetime,ID,Type,Status,From,To,Description,Amount,Currency\n"
        "2026-05-03T08:00:00Z,V3,Charge,Complete,Grace,Ada,Rent,($500.00),USD\n",
        encoding="utf-8",
    )
    since = SyncState(
        source_project="venmo_transactions_csv",
        source_entity_type="transaction",
        last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    all_result = VenmoTransactionsCsvAdapter(path=str(tmp_path)).ingest(entity_types=["transaction"])
    second_result = VenmoTransactionsCsvAdapter(path=str(tmp_path)).ingest(entity_types=["transaction"])
    since_result = VenmoTransactionsCsvAdapter(path=str(tmp_path)).ingest(since=since, entity_types=["transaction"])

    assert sorted(unit.metadata["transaction_id"] for unit in all_result.units) == ["V1", "V3"]
    assert [unit.source_id for unit in second_result.units] == [unit.source_id for unit in all_result.units]
    first = next(unit for unit in all_result.units if unit.metadata["transaction_id"] == "V1")
    assert first.metadata["counterparty"] == "Grace"
    assert first.metadata["source_file"] == "first.csv"
    assert first.metadata["url"] == "https://venmo.example/tx/V1"
    assert [unit.metadata["transaction_id"] for unit in since_result.units] == ["V3"]


def test_venmo_transactions_csv_emits_counterparty_aggregates_and_edges(tmp_path):
    export = tmp_path / "venmo.csv"
    export.write_text(
        "Date,Transaction ID,Type,Note,From,To,Amount\n"
        "2026-05-01,V1,Payment,Lunch,Ada,Grace,$12.50\n"
        "2026-05-03,V2,Charge,Rent,Grace,Ada,($500.00)\n"
        "2026-05-04,V3,Payment,Top up,,,$20.00\n",
        encoding="utf-8",
    )

    result = VenmoTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["transaction", "counterparty"])

    counterparties = [unit for unit in result.units if unit.source_entity_type == "counterparty"]
    assert len(counterparties) == 1
    counterparty = counterparties[0]
    assert counterparty.metadata["counterparty"] == "Ada"
    assert counterparty.metadata["transaction_count"] == 2
    assert counterparty.metadata["first_seen"] == "2026-05-01T00:00:00+00:00"
    assert counterparty.metadata["last_seen"] == "2026-05-03T00:00:00+00:00"
    assert counterparty.metadata["net_amount"] == -487.5
    assert counterparty.metadata["transaction_types"] == ["Charge", "Payment"]
    assert {(edge.from_unit_id, edge.to_unit_id, edge.metadata["relation_type"]) for edge in result.edges} == {
        ("venmo_transactions_csv:V1", counterparty.source_id, "transaction_counterparty"),
        ("venmo_transactions_csv:V2", counterparty.source_id, "transaction_counterparty"),
    }

    assert [unit.source_entity_type for unit in VenmoTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["counterparty"]).units] == ["counterparty"]
    assert VenmoTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["transaction"]).edges == []


def test_venmo_transactions_csv_is_registered():
    assert "venmo_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("venmo-transactions-csv"), VenmoTransactionsCsvAdapter)
