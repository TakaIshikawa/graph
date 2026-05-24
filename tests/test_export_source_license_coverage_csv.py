from __future__ import annotations

import csv
from io import StringIO
from types import SimpleNamespace

from graph.export import export_source_license_coverage_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_license_coverage_groups_normalized_rights_values():
    text = export_source_license_coverage_csv(
        [
            {"id": "s1", "metadata": {"license": "CC-BY-4.0"}},
            SimpleNamespace(id="s2", metadata={"rights": "All rights reserved"}),
            {"id": "s3", "metadata": {"terms_url": "https://example.test/terms"}},
            {"id": "s4"},
        ]
    )

    by_norm = {row["normalized_license"]: row for row in rows(text)}
    assert by_norm["cc-by"]["count"] == "1"
    assert by_norm["copyright"]["status"] == "restricted"
    assert by_norm["terms-url"]["status"] == "ambiguous"
    assert by_norm["missing"]["status"] == "missing"


def test_source_license_coverage_writes_path_metadata(tmp_path):
    path = tmp_path / "licenses.csv"
    stats = export_source_license_coverage_csv([{"id": "s1", "license": "MIT"}], path)

    assert stats["source_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
