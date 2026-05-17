from __future__ import annotations

from graph.adapters.quicken_transactions_csv import QuickenTransactionsCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject


def test_quicken_transactions_csv_ingests_register_rows(tmp_path):
    export = tmp_path / "quicken.csv"
    export.write_text(
        "Date,Transaction ID,Account,Check Number,Payee,Category,Tag,Memo,Cleared,Payment,Deposit,Balance\n"
        "2026-05-01,Q1,Checking,1001,Landlord,Rent,Home,May rent,R,1200,,2500\n",
        encoding="utf-8",
    )

    result = QuickenTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.QUICKEN_TRANSACTIONS_CSV
    assert unit.source_id == "quicken_transactions_csv:Q1"
    assert unit.metadata["timestamp"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["account"] == "Checking"
    assert unit.metadata["check_number"] == "1001"
    assert unit.metadata["payee"] == "Landlord"
    assert unit.metadata["category"] == "Rent"
    assert unit.metadata["tag"] == "Home"
    assert unit.metadata["memo"] == "May rent"
    assert unit.metadata["cleared"] == "R"
    assert unit.metadata["payment"] == 1200.0
    assert unit.metadata["amount"] == -1200.0
    assert unit.metadata["balance"] == 2500.0


def test_quicken_transactions_csv_handles_amount_and_deposit_columns(tmp_path):
    export = tmp_path / "quicken.csv"
    export.write_text(
        "Date,Payee,Amount,Deposit,Balance\n"
        "2026-05-02,Adjustment,-5,,2495\n"
        "2026-05-03,Payroll,,3000,5495\n",
        encoding="utf-8",
    )

    result = QuickenTransactionsCsvAdapter(path=str(export)).ingest()

    assert [unit.metadata["amount"] for unit in result.units] == [-5.0, 3000.0]


def test_quicken_transactions_csv_emits_account_aggregates_and_edges(tmp_path):
    export = tmp_path / "quicken.csv"
    export.write_text(
        "Date,Transaction ID,Account,Payee,Category,Amount\n"
        "2026-05-01,Q1,Checking,Landlord,Rent,-1200\n"
        "2026-05-03,Q2, checking ,Payroll,Income,3000\n"
        "2026-05-04,Q3,,Adjustment,Other,-5\n",
        encoding="utf-8",
    )

    result = QuickenTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["transaction", "account"])

    accounts = [unit for unit in result.units if unit.source_entity_type == "account"]
    assert len(accounts) == 1
    account = accounts[0]
    assert account.metadata["account"] == "Checking"
    assert account.metadata["transaction_count"] == 2
    assert account.metadata["first_seen"] == "2026-05-01T00:00:00+00:00"
    assert account.metadata["last_seen"] == "2026-05-03T00:00:00+00:00"
    assert account.metadata["net_amount"] == 1800.0
    assert account.metadata["categories"] == ["Income", "Rent"]
    assert {(edge.from_unit_id, edge.to_unit_id, edge.metadata["relation_type"]) for edge in result.edges} == {
        ("quicken_transactions_csv:Q1", account.source_id, "transaction_account"),
        ("quicken_transactions_csv:Q2", account.source_id, "transaction_account"),
    }

    assert [unit.source_entity_type for unit in QuickenTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["account"]).units] == ["account"]
    assert QuickenTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["transaction"]).edges == []


def test_quicken_transactions_csv_is_registered():
    assert "quicken_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("quicken-transactions-csv"), QuickenTransactionsCsvAdapter)
