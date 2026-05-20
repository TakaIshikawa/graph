from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_duplicate_url_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, title: str, source_project: str = "Project", content: str = "", metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=content,
        metadata=metadata or {},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_duplicate_url_csv_groups_normalized_duplicate_urls():
    text = export_source_duplicate_url_csv(
        [
            unit("b", title="Beta", source_project="Source B", metadata={"url": "HTTPS://Example.com/Path/"}),
            unit("a", title="Alpha", source_project="Source A", content="Read https://example.com/Path#top"),
            unit("c", title="Gamma", metadata={"url": "https://example.com/other"}),
        ]
    )

    assert rows(text) == [
        {
            "normalized_url": "https://example.com/Path",
            "domain": "example.com",
            "unit_count": "2",
            "sources": "Source A; Source B",
            "unit_ids": "a; b",
            "titles": "Alpha; Beta",
        }
    ]


def test_source_duplicate_url_csv_only_emits_urls_in_multiple_units():
    assert rows(
        export_source_duplicate_url_csv(
            [
                unit("a", title="Alpha", metadata={"url": "https://example.com/page", "canonical_url": "https://example.com/page/"}),
                unit("b", title="Beta", metadata={"url": "https://example.com/other"}),
            ]
        )
    ) == []


def test_source_duplicate_url_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "duplicates.csv"
    units = [
        unit("a", title="Alpha", metadata={"url": "https://example.com/page"}),
        unit("b", title="Beta", metadata={"url": "https://example.com/page/"}),
    ]

    expected = export_source_duplicate_url_csv(units)
    stats = export_source_duplicate_url_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["duplicate_url_count"] == 1
