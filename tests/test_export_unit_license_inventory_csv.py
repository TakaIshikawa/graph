from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_units_to_license_inventory_csv
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def unit(unit_id: str, *, metadata: dict | None = None, **overrides: object) -> KnowledgeUnit:
    data = {
        "id": unit_id,
        "source_project": "max",
        "source_id": f"source-{unit_id}",
        "source_entity_type": "note",
        "title": f"Title {unit_id}",
        "content": "Content",
        "metadata": metadata or {},
        "created_at": UNIT_TIME,
        "ingested_at": UNIT_TIME,
        "updated_at": UNIT_TIME,
    }
    data.update(overrides)
    return KnowledgeUnit(**data)


def test_export_units_to_license_inventory_csv_discovers_aliases_and_sorts_rows():
    text = export_units_to_license_inventory_csv(
        [
            {"id": "b", "title": "Beta", "source_project": "B", "metadata": {"Licence": "Custom Terms"}},
            unit("a", metadata={"license": ["CC-BY-4.0", ""], "license_url": "https://creativecommons.org/licenses/by/4.0/"}),
        ]
    )

    result = rows(text)
    assert [row["unit_id"] for row in result] == ["a", "b"]
    assert result[0]["license"] == "CC-BY-4.0"
    assert result[0]["license_family"] == "creative_commons"
    assert result[0]["license_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert result[1]["license"] == "Custom Terms"
    assert result[1]["license_family"] == "custom"


def test_export_units_to_license_inventory_csv_classifies_spdx_public_domain_and_copyright():
    result = rows(
        export_units_to_license_inventory_csv(
            [
                {"id": "mit", "title": "MIT", "metadata": {"spdx_id": "MIT"}},
                {"id": "pd", "title": "Public", "metadata": {"rights": "Public Domain"}},
                {"id": "copy", "title": "Copyright", "metadata": {"copyright_status": "All rights reserved"}},
            ]
        )
    )

    assert {row["unit_id"]: row["license_family"] for row in result} == {
        "copy": "copyrighted",
        "mit": "spdx",
        "pd": "public_domain",
    }


def test_export_units_to_license_inventory_csv_reports_missing_values_and_rights_fields():
    result = rows(
        export_units_to_license_inventory_csv(
            [
                {
                    "source_id": "source-1",
                    "title": "Untitled",
                    "metadata": {
                        "rights_holder": "Example Org",
                        "commercial_use": False,
                        "derivatives_allowed": ["yes", "with attribution"],
                    },
                }
            ]
        )
    )[0]

    assert result["unit_id"] == "source-1"
    assert result["source_project"] == "Unknown"
    assert result["license"] == ""
    assert result["license_family"] == "unknown"
    assert result["rights_holder"] == "Example Org"
    assert result["commercial_use"] == "False"
    assert result["derivatives_allowed"] == "with attribution; yes"


def test_export_units_to_license_inventory_csv_writes_path(tmp_path):
    path = tmp_path / "licenses.csv"

    stats = export_units_to_license_inventory_csv([{"id": "a", "title": "A", "license": "Apache-2.0"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["license_family"] == "spdx"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] > 0
