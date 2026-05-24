from __future__ import annotations

import csv
from io import StringIO
from types import SimpleNamespace

from graph.export import export_unit_currency_exposure_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_currency_exposure_groups_currency_and_amounts():
    text = export_unit_currency_exposure_csv(
        [
            {"id": "u1", "amount": "10.50", "metadata": {"currency": "usd"}},
            SimpleNamespace(id="u2", metadata={"amount": {"value": 5, "currency": "USD"}}),
            {"id": "u3", "metadata": {"price_currency": "eur", "price": "7"}},
        ]
    )

    by_currency = {row["currency"]: row for row in rows(text)}
    assert by_currency["USD"]["unit_count"] == "2"
    assert by_currency["USD"]["amount_count"] == "2"
    assert by_currency["USD"]["total_amount"] == "15.50"
    assert by_currency["EUR"]["total_amount"] == "7.00"


def test_unit_currency_exposure_writes_path_metadata(tmp_path):
    path = tmp_path / "currency.csv"
    stats = export_unit_currency_exposure_csv([{"id": "u1", "metadata": {"currency": "jpy"}}], path)

    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
