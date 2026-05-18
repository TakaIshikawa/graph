from __future__ import annotations

from graph.adapters.fidelity_account_activity_csv import FidelityAccountActivityCsvAdapter


def test_fidelity_account_activity_csv_ingests_brokerage_activity_rows(tmp_path):
    export = tmp_path / "fidelity.csv"
    export.write_text(
        "Run Date,Account,Action,Symbol,Description,Quantity,Price,Commission,Fees,Amount,Settlement Date,Reference Number\n"
        "04/01/2026,Z12345678,BUY,AAPL,APPLE INC,2,150.25,0.00,1.00,-301.50,04/03/2026,F100\n",
        encoding="utf-8",
    )

    result = FidelityAccountActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "fidelity_account_activity_csv"
    assert unit.source_entity_type == "account_activity"
    assert unit.source_id == "fidelity_account_activity_csv:F100"
    assert unit.metadata["run_date"] == "2026-04-01"
    assert unit.metadata["settlement_date"] == "2026-04-03"
    assert unit.metadata["account"] == "Z12345678"
    assert unit.metadata["action"] == "BUY"
    assert unit.metadata["symbol"] == "AAPL"
    assert unit.metadata["description"] == "APPLE INC"
    assert unit.metadata["quantity"] == 2.0
    assert unit.metadata["price"] == 150.25
    assert unit.metadata["commission"] == 0.0
    assert unit.metadata["fees"] == 1.0
    assert unit.metadata["amount"] == -301.5
    assert unit.metadata["reference_number"] == "F100"
    assert unit.metadata["source_file"] == "fidelity.csv"
    assert unit.tags[:3] == ["finance", "fidelity", "account_activity"]
    assert "BUY AAPL" in unit.title
    assert "APPLE INC" in unit.content


def test_fidelity_account_activity_csv_ingests_directories_and_digest_fallback(tmp_path):
    nested = tmp_path / "exports"
    nested.mkdir()
    export = nested / "fidelity.csv"
    export.write_text(
        "Run Date,Account,Action,Symbol,Description,Quantity,Price,Commission,Fees,Amount,Settlement Date\n"
        "2026-04-01,Z12345678,SELL,MSFT,MICROSOFT CORP,1,300,0,0,300,2026-04-03\n",
        encoding="utf-8",
    )

    first = FidelityAccountActivityCsvAdapter(path=str(tmp_path)).ingest().units[0]
    second = FidelityAccountActivityCsvAdapter(path=str(tmp_path)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("fidelity_account_activity_csv:")
    assert first.metadata["quantity"] == 1.0
    assert first.metadata["amount"] == 300.0
    assert first.metadata["source_file"] == "fidelity.csv"
