from __future__ import annotations

from graph.adapters.amex_transactions_csv import AmexTransactionsCsvAdapter


def test_amex_transactions_csv_ingests_representative_rows(tmp_path):
    export = tmp_path / "amex.csv"
    export.write_text(
        "Date,Description,Card Member,Account #,Amount,Extended Details,Appears On Your Statement As,Address,City/State,Zip Code,Country,Reference,Category\n"
        "04/01/2026,CAFE,ADA LOVELACE,1001,-4.50,Latte,CAFE CITY,1 Main St,San Francisco CA,94105,US,A123,Restaurant\n"
        "04/03/2026,ONLINE RETURN,ADA LOVELACE,1001,19.99,Refund,ONLINE SHOP,2 Market St,New York NY,10001,US,A124,Merchandise\n"
        "\n",
        encoding="utf-8",
    )

    result = AmexTransactionsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    purchase, refund = result.units
    assert purchase.source_project == "amex_transactions_csv"
    assert purchase.source_entity_type == "transaction"
    assert purchase.metadata["date"] == "2026-04-01"
    assert purchase.metadata["description"] == "CAFE"
    assert purchase.metadata["card_member"] == "ADA LOVELACE"
    assert purchase.metadata["account_number"] == "1001"
    assert purchase.metadata["amount"] == -4.5
    assert purchase.metadata["currency"] == "USD"
    assert purchase.metadata["extended_details"] == "Latte"
    assert purchase.metadata["statement_descriptor"] == "CAFE CITY"
    assert purchase.metadata["address"] == "1 Main St"
    assert purchase.metadata["city_state"] == "San Francisco CA"
    assert purchase.metadata["zip_code"] == "94105"
    assert purchase.metadata["country"] == "US"
    assert purchase.metadata["reference"] == "A123"
    assert purchase.metadata["category"] == "Restaurant"
    assert purchase.metadata["source_row"]["Appears On Your Statement As"] == "CAFE CITY"
    assert purchase.tags[:3] == ["finance", "transaction", "amex"]
    assert refund.metadata["amount"] == 19.99
    assert refund.metadata["reference"] == "A124"


def test_amex_transactions_csv_source_ids_are_deterministic(tmp_path):
    export = tmp_path / "amex.csv"
    export.write_text(
        "Date,Description,Card Member,Account #,Amount,Appears On Your Statement As,Reference,Category\n"
        "2026-04-01,CAFE,ADA LOVELACE,1001,($4.50),CAFE CITY,A123,Restaurant\n",
        encoding="utf-8",
    )

    first = AmexTransactionsCsvAdapter(path=str(export)).ingest().units[0]
    second = AmexTransactionsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("amex_transactions_csv:")
    assert first.metadata["amount"] == -4.5
