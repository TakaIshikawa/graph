from __future__ import annotations

from graph.export import export_units_to_anki_tsv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        tags=tags or [],
    )


def read_rows(path):
    return [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()]


def test_export_anki_tsv_writes_default_front_back_tags_and_stats(tmp_path):
    path = tmp_path / "deck.tsv"
    stats = export_units_to_anki_tsv(
        [
            unit(
                "unit-a",
                "Solar storage",
                "Batteries smooth evening demand.",
                ["Energy", "storage"],
            )
        ],
        path,
    )

    rows = read_rows(path)

    assert stats == {
        "path": str(path),
        "units_scanned": 1,
        "cards_exported": 1,
        "skipped_empty": 0,
    }
    assert rows == [
        [
            "Solar storage",
            "Batteries smooth evening demand. Source: max/insight (source-unit-a)",
            "energy storage",
        ]
    ]


def test_export_anki_tsv_normalizes_tags_deterministically(tmp_path):
    path = tmp_path / "deck.tsv"
    export_units_to_anki_tsv(
        [
            unit(
                "unit-a",
                "Title",
                "Content",
                ["AI/ML", "ai ml", "  Read Later  ", "read-later", "", "C++"],
            )
        ],
        path,
    )

    assert read_rows(path)[0][2] == "ai_ml c read-later read_later"


def test_export_anki_tsv_sanitizes_tabs_and_newlines_to_one_row_per_unit(tmp_path):
    path = tmp_path / "deck.tsv"
    export_units_to_anki_tsv(
        [
            unit("unit-a", "Question\tone", "Answer\none\r\nwith\ttab", ["tag\none"]),
            unit("unit-b", "Question two", "Answer two"),
        ],
        path,
        front_template="{title}\n{source_id}",
        back_template=lambda item: f"{item.content}\t{item.source_entity_type}",
    )

    text = path.read_text(encoding="utf-8")
    rows = read_rows(path)

    assert text.count("\n") == 2
    assert all(len(row) == 3 for row in rows)
    assert rows[0] == [
        "Question one source-unit-a",
        "Answer one with tab insight",
        "tag_one",
    ]


def test_export_anki_tsv_skips_empty_cards_and_can_omit_tags(tmp_path):
    path = tmp_path / "deck.tsv"
    stats = export_units_to_anki_tsv(
        [
            unit("unit-a", "", "Content"),
            unit("unit-b", "Title", ""),
            unit("unit-c", "Title", "Content", ["tag"]),
        ],
        path,
        include_tags=False,
    )

    assert stats == {
        "path": str(path),
        "units_scanned": 3,
        "cards_exported": 1,
        "skipped_empty": 2,
    }
    assert read_rows(path) == [["Title", "Content Source: max/insight (source-unit-c)"]]
