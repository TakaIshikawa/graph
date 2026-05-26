from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_frontmatter_key_audit_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_frontmatter_key_audit_counts_unique_and_duplicate_keys():
    result = rows(export_units_to_frontmatter_key_audit_csv([{"id": "u", "content": "---\ntitle: A\ntags: x\ntitle: B\n---\nbody"}]))[0]

    assert result == {"unit_id": "u", "has_frontmatter": "true", "key_count": "2", "keys": "tags; title", "duplicate_key_count": "1", "malformed_frontmatter": "false"}


def test_frontmatter_key_audit_reports_malformed_and_ignores_non_initial(tmp_path):
    output = tmp_path / "frontmatter.csv"
    result = export_units_to_frontmatter_key_audit_csv([{"id": "a", "content": "body\n---\ntitle: no"}, {"id": "b", "content": "---\ntitle: yes"}], output)

    assert result["rows_exported"] == 2
    parsed = {row["unit_id"]: row for row in rows(output.read_text(encoding="utf-8"))}
    assert parsed["a"]["has_frontmatter"] == "false"
    assert parsed["b"]["malformed_frontmatter"] == "true"
