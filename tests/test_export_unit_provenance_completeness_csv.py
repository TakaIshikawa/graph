from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_unit_provenance_completeness_csv
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: str | None = "Project",
    source_id: str | None = None,
    source_entity_type: str | None = "note",
    metadata: dict | None = None,
    created_at=None,
):
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}" if source_id is None else source_id,
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        tags=[],
        created_at=created_at,
        updated_at=None,
        ingested_at=None,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_provenance_completeness_csv_empty_input_has_header_only():
    assert export_unit_provenance_completeness_csv([]) == (
        "unit_id,source_project,source_entity_type,score,missing_fields,provenance_url\n"
    )


def test_unit_provenance_completeness_csv_scores_aliases_and_dates():
    text = export_unit_provenance_completeness_csv(
        [
            unit(
                "complete",
                metadata={"permalink": "https://example.test/a", "account": "ada"},
                created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            ),
            unit("missing", source_project=None, source_id="", metadata={"source_url": "https://example.test/b"}),
            unit("partial", metadata={"author": "Grace", "imported_at": "2026-05-02"}),
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "missing",
            "source_project": "Unknown",
            "source_entity_type": "note",
            "score": "0.33",
            "missing_fields": "source_project; source_id; author_or_account; source_date",
            "provenance_url": "https://example.test/b",
        },
        {
            "unit_id": "partial",
            "source_project": "Project",
            "source_entity_type": "note",
            "score": "0.83",
            "missing_fields": "provenance_url",
            "provenance_url": "",
        },
        {
            "unit_id": "complete",
            "source_project": "Project",
            "source_entity_type": "note",
            "score": "1.00",
            "missing_fields": "",
            "provenance_url": "https://example.test/a",
        },
    ]


def test_unit_provenance_completeness_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "provenance.csv"
    units = [unit("a", metadata={"url": "https://example.test", "author": "Ada"})]

    expected = export_unit_provenance_completeness_csv(units)
    stats = export_unit_provenance_completeness_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {"path": str(path), "unit_count": 1, "rows_exported": 1, "bytes_written": path.stat().st_size}
