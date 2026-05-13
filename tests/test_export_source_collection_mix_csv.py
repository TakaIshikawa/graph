from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_collection_mix_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str | None = SourceProject.MAX,
    source_entity_type: str | None = "note",
    metadata: object | None = None,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata={} if metadata is None else metadata,
        tags=[] if tags is None else tags,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_collection_mix_csv_groups_by_collection_and_source_project():
    text = export_source_collection_mix_csv(
        [
            unit("a", metadata={"collection": "Inbox"}, tags=["ai", "review"]),
            unit("b", metadata={"collections": ["Inbox", "Research"]}, tags=["ai"]),
            unit(
                "c",
                source_project=SourceProject.PINBOARD,
                source_entity_type="bookmark",
                metadata={"folder": "Inbox"},
                tags=["web"],
            ),
        ]
    )

    assert text.splitlines()[0] == "collection,source_project,unit_count,source_entity_types,top_tags"
    assert rows(text) == [
        {
            "collection": "Inbox",
            "source_project": "max",
            "unit_count": "2",
            "source_entity_types": "note:2",
            "top_tags": "ai:2; review:1",
        },
        {
            "collection": "Inbox",
            "source_project": "pinboard",
            "unit_count": "1",
            "source_entity_types": "bookmark:1",
            "top_tags": "web:1",
        },
        {
            "collection": "Research",
            "source_project": "max",
            "unit_count": "1",
            "source_entity_types": "note:1",
            "top_tags": "ai:1",
        },
    ]


def test_source_collection_mix_csv_groups_units_without_collection_as_unassigned():
    text = export_source_collection_mix_csv(
        [
            unit("a", source_project=None, source_entity_type=None, metadata={}),
            unit("b", source_project="", source_entity_type="", metadata={"collection": ""}),
        ]
    )

    assert rows(text) == [
        {
            "collection": "Unassigned",
            "source_project": "Unknown",
            "unit_count": "2",
            "source_entity_types": "Unknown:2",
            "top_tags": "",
        }
    ]


def test_source_collection_mix_csv_custom_collection_keys_override_defaults():
    text = export_source_collection_mix_csv(
        [
            unit("a", metadata={"collection": "Default", "space": "Custom"}),
            unit("b", metadata={"space": ["Custom", "Archive"]}),
        ],
        collection_keys=["space"],
    )

    assert rows(text) == [
        {
            "collection": "Archive",
            "source_project": "max",
            "unit_count": "1",
            "source_entity_types": "note:1",
            "top_tags": "",
        },
        {
            "collection": "Custom",
            "source_project": "max",
            "unit_count": "2",
            "source_entity_types": "note:2",
            "top_tags": "",
        },
    ]


def test_source_collection_mix_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "collection-mix.csv"
    units = [unit("a", metadata={"board": "Roadmap"})]

    expected = export_source_collection_mix_csv(units)
    stats = export_source_collection_mix_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "collection_key_count": 6,
        "bytes_written": path.stat().st_size,
    }
