from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.etrade_transactions_csv import EtradeTransactionsCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_etrade_transactions_csv_ingests_transaction_rows(tmp_path):
    export = tmp_path / "etrade.csv"
    export.write_text(
        "TransactionDate,Transaction Type,Security Type,Symbol,Quantity,Amount,Description\n"
        "2026-05-01,BUY,Equity,AAPL,2,-300,APPLE INC\n",
        encoding="utf-8",
    )

    result = EtradeTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.ETRADE_TRANSACTIONS_CSV
    assert unit.source_entity_type == "brokerage_transaction"
    assert unit.metadata["timestamp"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["transaction_type"] == "BUY"
    assert unit.metadata["security_type"] == "Equity"
    assert unit.metadata["symbol"] == "AAPL"
    assert unit.metadata["quantity"] == 2.0
    assert unit.metadata["amount"] == -300.0
    assert unit.metadata["description"] == "APPLE INC"
    assert unit.metadata["source_file"] == "etrade.csv"


def test_etrade_transactions_csv_supports_header_variants_filters_ids_and_registry(tmp_path):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    (export_dir / "old.csv").write_text("Transaction Date,TransactionType,SecurityType,Amount\n2026-05-01,OLD,Equity,1\n", encoding="utf-8")
    (export_dir / "new.csv").write_text("Transaction Date,Transaction ID,TransactionType,SecurityType,Amount\n2026-05-03,E1,NEW,Mutual Fund,2\n", encoding="utf-8")

    adapter = EtradeTransactionsCsvAdapter(path=str(export_dir))
    since = SyncState(source_project="etrade_transactions_csv", source_entity_type="brokerage_transaction", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = adapter.ingest(since=since)
    assert [unit.metadata["transaction_type"] for unit in result.units] == ["NEW"]
    assert result.units[0].metadata["security_type"] == "Mutual Fund"
    assert result.units[0].source_id == "etrade_transactions_csv:E1"
    assert adapter.ingest(entity_types=["transaction"]).units == []
    assert "etrade_transactions_csv" in list_adapters()
    assert isinstance(get_adapter("etrade-transactions-csv"), EtradeTransactionsCsvAdapter)
