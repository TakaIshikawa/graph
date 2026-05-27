from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_local_file_reference_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_units_to_local_file_reference_csv_detects_local_targets(tmp_path):
    (tmp_path / "doc.pdf").write_text("x")
    rows = _rows(export_units_to_local_file_reference_csv([{"id": "u", "content": "[doc](doc.pdf)\n![img](/tmp/a.png)\n[web](https://example.com/a)\nfile:///tmp/raw.txt"}], base_path=tmp_path))

    assert [(row["path"], row["scheme"], row["extension"], row["exists"]) for row in rows] == [
        ("doc.pdf", "", "pdf", "true"),
        ("/tmp/a.png", "", "png", "false"),
        ("/tmp/raw.txt", "file", "txt", "false"),
    ]
