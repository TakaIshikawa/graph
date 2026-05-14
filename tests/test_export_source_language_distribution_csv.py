from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_source_language_distribution_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def unit(unit_id: str, *, source_entity_type: str = "note", metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content="Content",
        metadata=metadata or {},
        tags=[],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_language_distribution_csv_empty_input_returns_header():
    assert export_source_language_distribution_csv([]) == (
        "source_project,source_entity_type,language_code,language_label,unit_count,percent_of_group\n"
    )


def test_export_source_language_distribution_csv_groups_normalized_language_codes():
    text = export_source_language_distribution_csv(
        [
            unit("a", metadata={"language": "en-US"}),
            unit("b", metadata={"lang": "EN_gb"}),
            unit("c", metadata={"locale": "ja_JP"}),
        ]
    )

    result = {row["language_code"]: row for row in rows(text)}
    assert result["en"]["language_label"] == "English"
    assert result["en"]["unit_count"] == "2"
    assert result["en"]["percent_of_group"] == "66.67"
    assert result["ja"]["language_label"] == "Japanese"


def test_export_source_language_distribution_csv_reports_unknown_for_missing_blank_language():
    text = export_source_language_distribution_csv(
        [
            unit("a", metadata={}),
            unit("b", metadata={"content_language": " "}),
            unit("c", source_entity_type="task", metadata={"source_language": "fr"}),
        ]
    )

    result = {(row["source_entity_type"], row["language_code"]): row for row in rows(text)}
    assert result[("note", "unknown")]["language_label"] == "Unknown"
    assert result[("note", "unknown")]["unit_count"] == "2"
    assert result[("task", "fr")]["unit_count"] == "1"


def test_export_source_language_distribution_csv_path_mode(tmp_path):
    path = tmp_path / "languages.csv"
    stats = export_source_language_distribution_csv([unit("a", metadata={"language": "es-MX"})], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["language_code"] == "es"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
