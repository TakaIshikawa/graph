from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.export import export_units_to_csl_json
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(unit_id: str, *, metadata: dict | None = None, content: str = "Abstract body.") -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.CSL_JSON,
        source_id=f"source-{unit_id}",
        source_entity_type="publication",
        title=f"Title {unit_id}",
        content=content,
        content_type=ContentType.FINDING,
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
        metadata=metadata or {},
    )


def test_export_units_to_csl_json_maps_bibliographic_fields():
    records = json.loads(
        export_units_to_csl_json(
            [
                unit(
                    "a",
                    metadata={
                        "title": "Graph Citations",
                        "authors": ["Smith, Ada", {"family": "Doe", "given": "Grace"}],
                        "issued": "2025-04-24",
                        "doi": "10.1234/example",
                        "isbn": "978-1-234",
                        "url": "https://example.test/paper",
                        "journal": "Knowledge Review",
                        "publisher": "Example Press",
                        "type": "article",
                        "abstract": "Mapped abstract.",
                    },
                )
            ]
        )
    )

    assert records == [
        {
            "id": "a",
            "type": "article-journal",
            "title": "Graph Citations",
            "author": [
                {"family": "Smith", "given": "Ada"},
                {"family": "Doe", "given": "Grace"},
            ],
            "issued": {"date-parts": [[2025, 4, 24]]},
            "DOI": "10.1234/example",
            "ISBN": "978-1-234",
            "URL": "https://example.test/paper",
            "container-title": "Knowledge Review",
            "publisher": "Example Press",
            "abstract": "Mapped abstract.",
        }
    ]


def test_export_units_to_csl_json_minimally_serializes_notes_and_writes_file(tmp_path):
    path = tmp_path / "units.json"

    text = export_units_to_csl_json(unit("a", content="Fallback abstract."), path)

    assert path.read_text(encoding="utf-8") == text
    assert json.loads(text)[0]["abstract"] == "Fallback abstract."
