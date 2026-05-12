from __future__ import annotations

import csv
from io import StringIO

from graph.export.source_content_mix_csv import export_source_content_mix_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    source_project: SourceProject | str | None,
    content_type: ContentType | str | None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        content_type=content_type,
        metadata={},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_content_mix_groups_by_source_and_content_type():
    text = export_source_content_mix_csv(
        [
            unit("b", SourceProject.READWISE, ContentType.INSIGHT),
            unit("a", SourceProject.READWISE, ContentType.FINDING),
            unit("c", SourceProject.READWISE, ContentType.INSIGHT),
            unit("d", SourceProject.MAX, ContentType.METADATA),
        ]
    )

    assert text.splitlines()[0] == (
        "source_project,content_type,unit_count,source_unit_count,source_percent"
    )
    assert rows(text) == [
        {
            "source_project": "max",
            "content_type": "metadata",
            "unit_count": "1",
            "source_unit_count": "1",
            "source_percent": "100.00",
        },
        {
            "source_project": "readwise",
            "content_type": "insight",
            "unit_count": "2",
            "source_unit_count": "3",
            "source_percent": "66.67",
        },
        {
            "source_project": "readwise",
            "content_type": "finding",
            "unit_count": "1",
            "source_unit_count": "3",
            "source_percent": "33.33",
        },
    ]


def test_source_content_mix_normalizes_blank_values_to_unknown():
    text = export_source_content_mix_csv(
        [
            unit("a", None, None),
            unit("b", "  ", " \n\t "),
            unit("c", "  Web\nNotes  ", "Long\nForm"),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Unknown",
            "content_type": "Unknown",
            "unit_count": "2",
            "source_unit_count": "2",
            "source_percent": "100.00",
        },
        {
            "source_project": "Web Notes",
            "content_type": "Long Form",
            "unit_count": "1",
            "source_unit_count": "1",
            "source_percent": "100.00",
        },
    ]


def test_source_content_mix_is_deterministic_for_reversed_input():
    units = [
        unit("a", "Source B", "zeta"),
        unit("b", "Source A", "beta"),
        unit("c", "Source A", "alpha"),
        unit("d", "Source A", "alpha"),
    ]

    assert export_source_content_mix_csv(units) == export_source_content_mix_csv(reversed(units))
    assert [row["content_type"] for row in rows(export_source_content_mix_csv(units))] == [
        "alpha",
        "beta",
        "zeta",
    ]


def test_source_content_mix_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "content-mix.csv"
    units = [unit("a", "Source A", "note"), unit("b", "Source A", "clip")]

    expected = export_source_content_mix_csv(units)
    stats = export_source_content_mix_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "rows_exported": 2,
        "bytes_written": path.stat().st_size,
    }
