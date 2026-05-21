from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_unit_citation_readiness_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def unit(unit_id: str, **overrides: object) -> KnowledgeUnit:
    values = {
        "id": unit_id,
        "source_project": SourceProject.MAX,
        "source_id": f"source-{unit_id}",
        "source_entity_type": "item",
        "title": f"Title {unit_id}",
        "content": "Content",
        "metadata": {},
        "tags": [],
        "created_at": UNIT_TIME,
        "ingested_at": UNIT_TIME,
        "updated_at": UNIT_TIME,
    }
    values.update(overrides)
    return KnowledgeUnit.model_construct(**values)


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_citation_readiness_csv_empty_input_returns_header():
    assert export_unit_citation_readiness_csv([]) == (
        "unit_id,source_project,has_title,has_creator,has_date,has_url,has_doi,has_isbn,missing_fields,readiness_score\n"
    )


def test_export_unit_citation_readiness_csv_scores_mapping_and_model_units():
    text = export_unit_citation_readiness_csv(
        [
            {"id": "b", "source_project": "web", "title": "", "metadata": {"author": "Ada", "doi": "10.1/x"}},
            unit("a", metadata={"authors": ["Ada"], "published": "2026-01-01", "url": "https://example.test"}),
        ]
    )

    result = {row["unit_id"]: row for row in rows(text)}
    assert list(result) == ["a", "b"]
    assert result["a"]["has_creator"] == "true"
    assert result["a"]["has_url"] == "true"
    assert result["a"]["readiness_score"] == "1.00"
    assert result["b"]["has_doi"] == "true"
    assert result["b"]["missing_fields"] == "title;date"


def test_export_unit_citation_readiness_csv_path_mode(tmp_path):
    path = tmp_path / "readiness.csv"
    stats = export_unit_citation_readiness_csv([unit("a", metadata={"author": "Ada", "url": "x"})], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["unit_id"] == "a"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size

