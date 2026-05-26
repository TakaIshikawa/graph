from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_source_path_depth_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_path_depth_handles_local_paths_urls_and_missing_values():
    result = rows(
        export_units_to_source_path_depth_csv(
            [
                {"id": "local", "metadata": {"source_path": "notes/project/file.md"}},
                {"id": "url", "source_url": "https://example.com/a/b/page.html?x=1"},
                {"id": "none"},
            ]
        )
    )

    parsed = {row["unit_id"]: row for row in result}
    assert parsed["local"]["depth"] == "2"
    assert parsed["local"]["basename"] == "file.md"
    assert parsed["local"]["extension"] == ".md"
    assert parsed["url"]["is_url"] == "true"
    assert parsed["url"]["basename"] == "page.html"
    assert parsed["none"]["source_path"] == ""


def test_source_path_depth_writes_metadata(tmp_path):
    output = tmp_path / "paths.csv"
    result = export_units_to_source_path_depth_csv([{"id": "u", "filename": "README"}], output)

    assert result["unit_count"] == 1
    assert result["bytes_written"] == output.stat().st_size
    assert rows(output.read_text(encoding="utf-8"))[0]["basename"] == "README"
