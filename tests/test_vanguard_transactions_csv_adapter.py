from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.vanguard_transactions_csv import VanguardTransactionsCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_vanguard_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "vanguard-transactions.csv"
    export.write_text(
        "Trade Date,Settlement Date,Account,Investment Name,Symbol,Transaction Type,Shares,Share Price,Principal Amount,Commission Fees,Net Amount,Transaction ID\n"
        "2026-05-01,2026-05-03,Brokerage,Vanguard Total Stock,VTSAX,Buy,2,150,-300,-1,-301,VT-1\n",
        encoding="utf-8",
    )

    result = VanguardTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "vanguard_transactions_csv"
    assert unit.source_entity_type == "investment_transaction"
    assert unit.source_id == "vanguard_transactions_csv:VT-1"
    assert unit.title == "Buy VTSAX (-301)"
    assert unit.content_type == ContentType.METADATA
    assert unit.metadata["transaction_id"] == "VT-1"
    assert unit.metadata["trade_date"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["settlement_date"] == "2026-05-03T00:00:00+00:00"
    assert unit.metadata["account"] == "Brokerage"
    assert unit.metadata["investment_name"] == "Vanguard Total Stock"
    assert unit.metadata["symbol"] == "VTSAX"
    assert unit.metadata["transaction_type"] == "Buy"
    assert unit.metadata["shares"] == 2.0
    assert unit.metadata["share_price"] == 150.0
    assert unit.metadata["principal_amount"] == -300.0
    assert unit.metadata["commission_fees"] == -1.0
    assert unit.metadata["net_amount"] == -301.0
    assert unit.metadata["source_file"] == "vanguard-transactions.csv"
    assert "Account: Brokerage" in unit.content
    assert "Commission fees: -1.0" in unit.content
    assert unit.tags == ["vanguard", "investment_transaction", "Buy", "VTSAX", "Brokerage"]


def test_vanguard_transactions_csv_accepts_directory_and_deterministic_fallback_ids(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "one.csv").write_text(
        "Trade Date,Account,Investment Name,Symbol,Transaction Type,Net Amount\n"
        "2026-05-01,IRA,Vanguard 500 Index,VFIAX,Dividend,10.50\n",
        encoding="utf-8",
    )
    (export_dir / "two.csv").write_text(
        "Trade Date,Account,Investment Name,Symbol,Transaction Type,Net Amount\n"
        "2026-05-02,IRA,Vanguard Total Bond,VBTLX,Reinvestment,-20.25\n",
        encoding="utf-8",
    )
    (export_dir / "ignored.txt").write_text("Trade Date,Transaction Type\n2026-05-03,Ignored\n", encoding="utf-8")

    first = VanguardTransactionsCsvAdapter(path=str(export_dir)).ingest()
    second = VanguardTransactionsCsvAdapter(path=str(export_dir)).ingest()

    assert [unit.metadata["symbol"] for unit in first.units] == ["VFIAX", "VBTLX"]
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert all(unit.source_id.startswith("vanguard_transactions_csv:") for unit in first.units)


def test_vanguard_transactions_csv_supports_since_and_entity_filters(tmp_path):
    export = tmp_path / "vanguard-transactions.csv"
    export.write_text(
        "Trade Date,Transaction ID,Transaction Type,Net Amount\n"
        "2026-05-01,OLD,Old,1\n"
        "2026-05-03,NEW,New,2\n",
        encoding="utf-8",
    )

    filtered = VanguardTransactionsCsvAdapter(path=str(export)).ingest(
        since=SyncState(
            source_project="vanguard_transactions_csv",
            source_entity_type="investment_transaction",
            last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
        )
    )
    wrong_entity = VanguardTransactionsCsvAdapter(path=str(export)).ingest(entity_types=["transaction"])

    assert [unit.source_id for unit in filtered.units] == ["vanguard_transactions_csv:NEW"]
    assert wrong_entity.units == []
    assert wrong_entity.edges == []


def test_vanguard_transactions_csv_skips_empty_rows(tmp_path):
    export = tmp_path / "vanguard-transactions.csv"
    export.write_text(
        "Trade Date,Transaction Type,Net Amount\n"
        ",,\n",
        encoding="utf-8",
    )

    assert VanguardTransactionsCsvAdapter(path=str(export)).ingest().units == []
