from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_content_language_hint_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, title: str = "Title", content: str = "", metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project="Project",
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=content,
        metadata=metadata or {},
        tags=[],
        created_at=None,
        updated_at=None,
        ingested_at=None,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_content_language_hint_csv_empty_input_has_header_only():
    assert export_unit_content_language_hint_csv([]) == (
        "source,unit_id,title,language_hint,confidence,evidence\n"
    )


def test_unit_content_language_hint_csv_detects_supported_languages():
    text = export_unit_content_language_hint_csv(
        [
            unit("en", content="This is the plan and the work for the graph"),
            unit("ja", title="日本語", content="これは日本語の文章です。検索と知識のメモです。"),
            unit("es", content="El plan de la busqueda y de la lectura para el equipo"),
            unit("fr", content="Le plan de la recherche et de la lecture pour les equipes"),
        ]
    )

    assert [row["language_hint"] for row in rows(text)] == ["English", "Spanish", "French", "Japanese"]
    assert all(row["confidence"] != "0.00" for row in rows(text))


def test_unit_content_language_hint_csv_uses_metadata_text_and_unknown_for_short_text():
    text = export_unit_content_language_hint_csv(
        [
            unit("meta", title="", content="", metadata={"summary": "The metadata has the evidence and the context"}),
            unit("short", title="", content="tiny"),
        ]
    )

    assert rows(text) == [
        {
            "source": "Project",
            "unit_id": "meta",
            "title": "",
            "language_hint": "English",
            "confidence": "1.00",
            "evidence": "stopwords=4",
        },
        {
            "source": "Project",
            "unit_id": "short",
            "title": "",
            "language_hint": "Unknown",
            "confidence": "0.00",
            "evidence": "too_short",
        },
    ]


def test_unit_content_language_hint_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "languages.csv"
    units = [unit("a", content="This is the graph and the note")]

    expected = export_unit_content_language_hint_csv(units)
    stats = export_unit_content_language_hint_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {"path": str(path), "unit_count": 1, "rows_exported": 1, "bytes_written": path.stat().st_size}
