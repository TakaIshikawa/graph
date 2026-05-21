from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_link_rot_risk_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_link_rot_risk_csv_groups_by_source_project():
    text = export_source_link_rot_risk_csv(
        [
            {"id": "a", "source_project": "A", "metadata": {"url": "https://example.test", "last_checked": "2026-01-01"}},
            {"id": "b", "source_project": "A", "metadata": {"url": "ftp://example.test/file"}},
            {"id": "c", "source_project": "", "metadata": {}},
        ],
        reference_date="2026-05-01",
    )

    result = {row["source_project"]: row for row in rows(text)}
    assert result["A"]["unit_count"] == "2"
    assert result["A"]["total_urls"] == "2"
    assert result["A"]["non_http_url_count"] == "1"
    assert result["Unknown"]["missing_url_count"] == "1"


def test_export_source_link_rot_risk_csv_discovers_url_aliases_and_archives():
    row = rows(
        export_source_link_rot_risk_csv(
            [{"source_project": "A", "metadata": {"links": ["https://web.archive.org/web/x"]}}],
            reference_date="2026-05-01",
        )
    )[0]

    assert row["total_urls"] == "1"
    assert row["archived_url_count"] == "1"

