from __future__ import annotations

import csv
from io import StringIO
from types import SimpleNamespace

from graph.export import export_source_ssl_expiry_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_ssl_expiry_csv_buckets_sources_and_invalid_dates():
    text = export_source_ssl_expiry_csv(
        [
            {"id": "expired", "url": "https://expired.test", "metadata": {"ssl_expires_at": "2024-12-31"}},
            {"id": "soon", "metadata": {"fetch": {"tls_not_after": "2025-01-20T00:00:00Z"}}},
            SimpleNamespace(id="valid", url="https://valid.test", metadata={"cert_expires_at": "2025-03-01"}),
            {"id": "bad", "metadata": {"certificate_expiry": "next week"}},
            {"id": "missing"},
        ],
        reference_date="2025-01-01",
        warning_days=30,
    )

    by_id = {row["source_id"]: row for row in rows(text)}
    assert by_id["expired"]["status"] == "expired"
    assert by_id["soon"]["days_until_expiry"] == "19"
    assert by_id["soon"]["status"] == "expiring_soon"
    assert by_id["valid"]["status"] == "valid"
    assert by_id["bad"]["status"] == "invalid"
    assert by_id["missing"]["status"] == "missing"


def test_source_ssl_expiry_csv_writes_path_metadata(tmp_path):
    path = tmp_path / "nested" / "ssl.csv"
    stats = export_source_ssl_expiry_csv([{"id": "s1", "metadata": {"not_after": "2025-01-02"}}], path, reference_date="2025-01-01")

    assert stats["path"] == str(path)
    assert stats["source_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
    assert rows(path.read_text())[0]["expiry_date"] == "2025-01-02"
