from __future__ import annotations

import csv
from dataclasses import dataclass, field
from io import StringIO

from graph.export.unit_source_attribution_csv import export_unit_source_attribution_csv
from graph.types.models import KnowledgeUnit


@dataclass
class UnitLike:
    id: str
    title: str
    source_project: str | None = None
    source_id: str | None = None
    metadata: dict = field(default_factory=dict)


def unit(
    unit_id: str,
    *,
    source_project: str | None = "Project A",
    source_id: str | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}" if source_id is None else source_id,
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_source_attribution_csv_empty_input_has_header_only():
    assert export_unit_source_attribution_csv([]) == (
        "unit_id,title,source_project,attribution_score,missing_fields,present_fields,"
        "source_url,author,citation\n"
    )


def test_unit_source_attribution_csv_reports_incomplete_and_excludes_complete_units():
    text = export_unit_source_attribution_csv(
        [
            unit(
                "complete",
                metadata={
                    "url": "https://example.test/a",
                    "author": "Ada Lovelace",
                    "citation": "Example A",
                },
            ),
            unit("missing", source_project=None, source_id="", metadata={"url": "https://example.test/b"}),
            unit("sparse", metadata={"creator": "Grace Hopper"}),
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "missing",
            "title": "Title missing",
            "source_project": "Unknown",
            "attribution_score": "0.20",
            "missing_fields": "source_project; source_id; author; provenance",
            "present_fields": "source_url",
            "source_url": "https://example.test/b",
            "author": "",
            "citation": "",
        },
        {
            "unit_id": "sparse",
            "title": "Title sparse",
            "source_project": "Project A",
            "attribution_score": "0.60",
            "missing_fields": "source_url; provenance",
            "present_fields": "source_project; source_id; author",
            "source_url": "",
            "author": "Grace Hopper",
            "citation": "",
        },
    ]


def test_unit_source_attribution_csv_accepts_mappings_objects_and_file_provenance():
    text = export_unit_source_attribution_csv(
        [
            {
                "id": "map",
                "title": "Mapping",
                "source_project": "Project B",
                "source_id": "m-1",
                "source_url": "https://example.test/map",
                "metadata": {"Author": ["Beta Writer", "Alpha Writer"], "file_path": "/tmp/source.md"},
            },
            UnitLike("object", "Object", "Project C", "o-1", {"imported_from": "bear", "creator": "Creator"}),
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "object",
            "title": "Object",
            "source_project": "Project C",
            "attribution_score": "0.80",
            "missing_fields": "source_url",
            "present_fields": "source_project; source_id; author; provenance",
            "source_url": "",
            "author": "Creator",
            "citation": "",
        }
    ]


def test_unit_source_attribution_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "unit-source-attribution.csv"
    units = [unit("a", metadata={"author": "Ada"})]

    expected = export_unit_source_attribution_csv(units)
    stats = export_unit_source_attribution_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "weak_unit_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
