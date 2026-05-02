from __future__ import annotations

import pytest

from graph.export import export_concept_map_markdown
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(
    unit_id: str,
    title: str,
    *,
    tags: list[str] | None = None,
    source_project: SourceProject | str = SourceProject.MAX,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=f"{title} content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags or [],
    )


def edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation = EdgeRelation.RELATES_TO,
    *,
    weight: float = 1.0,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=EdgeSource.INFERRED,
        weight=weight,
    )


def test_export_concept_map_groups_by_tags_and_lists_strongest_relationships():
    text = export_concept_map_markdown(
        [
            unit("beta", "Beta", tags=["storage"]),
            unit("alpha", "Alpha", tags=["solar", "storage"]),
        ],
        [
            edge("weak", "alpha", "beta", EdgeRelation.REFERENCES, weight=0.4),
            edge("strong", "alpha", "beta", EdgeRelation.BUILDS_ON, weight=0.9),
        ],
    )

    assert text == (
        "# Concept Map\n"
        "\n"
        "## solar\n"
        "\n"
        "- Alpha (`alpha`)\n"
        "  - builds_on -> Beta (`beta`) - weight 0.9\n"
        "  - references -> Beta (`beta`) - weight 0.4\n"
        "\n"
        "## storage\n"
        "\n"
        "- Alpha (`alpha`)\n"
        "  - builds_on -> Beta (`beta`) - weight 0.9\n"
        "  - references -> Beta (`beta`) - weight 0.4\n"
        "- Beta (`beta`)\n"
        "  - builds_on <- Alpha (`alpha`) - weight 0.9\n"
        "  - references <- Alpha (`alpha`) - weight 0.4\n"
    )


def test_export_concept_map_groups_by_source():
    text = export_concept_map_markdown(
        [
            unit("csv", "CSV Note", source_project=SourceProject.CSV),
            unit("max", "Max Note", source_project=SourceProject.MAX),
        ],
        [],
        group_by="source",
    )

    assert text.splitlines() == [
        "# Concept Map",
        "",
        "## csv",
        "",
        "- CSV Note (`csv`)",
        "  - _No linked concepts._",
        "",
        "## max",
        "",
        "- Max Note (`max`)",
        "  - _No linked concepts._",
    ]


def test_export_concept_map_uses_metadata_titles_and_identifier_fallbacks():
    text = export_concept_map_markdown(
        [
            unit("fallback", "", tags=["review"]),
            unit("metadata", "", tags=["review"], metadata={"title": "Metadata Title"}),
            unit("label", "Ignored Title", tags=["review"], metadata={"label": "Curated Label"}),
        ],
        [edge("link", "fallback", "metadata", weight=0.7)],
    )

    assert "- fallback (`fallback`)" in text
    assert "- Metadata Title (`metadata`)" in text
    assert "- Curated Label (`label`)" in text
    assert "relates_to -> Metadata Title (`metadata`) - weight 0.7" in text


def test_export_concept_map_respects_max_links_per_unit():
    text = export_concept_map_markdown(
        [
            unit("alpha", "Alpha", tags=["map"]),
            unit("beta", "Beta", tags=["map"]),
            unit("gamma", "Gamma", tags=["map"]),
        ],
        [
            edge("third", "alpha", "gamma", EdgeRelation.REFERENCES, weight=0.1),
            edge("first", "alpha", "beta", EdgeRelation.BUILDS_ON, weight=0.9),
            edge("second", "alpha", "gamma", EdgeRelation.CHALLENGES, weight=0.5),
        ],
        max_links_per_unit=2,
    )

    alpha_section = text.split("- Alpha (`alpha`)\n", maxsplit=1)[1].split("- Beta", maxsplit=1)[0]

    assert "builds_on -> Beta (`beta`) - weight 0.9" in alpha_section
    assert "challenges -> Gamma (`gamma`) - weight 0.5" in alpha_section
    assert "references -> Gamma (`gamma`) - weight 0.1" not in alpha_section


def test_export_concept_map_output_is_deterministic():
    units = [
        unit("beta", "Beta", tags=["zeta", "alpha"]),
        unit("alpha", "Alpha", tags=["alpha"]),
    ]
    relationships = [
        edge("b", "beta", "alpha", EdgeRelation.REFERENCES, weight=0.3),
        edge("a", "alpha", "beta", EdgeRelation.BUILDS_ON, weight=0.3),
    ]

    first = export_concept_map_markdown(units, relationships)
    second = export_concept_map_markdown(reversed(units), reversed(relationships))

    assert first == second


def test_export_concept_map_is_importable_from_graph_export():
    from graph.export import export_concept_map_markdown as imported

    assert imported is export_concept_map_markdown


@pytest.mark.parametrize("group_by", ["tags", "project", ""])
def test_export_concept_map_validates_grouping_mode(group_by):
    with pytest.raises(ValueError, match="group_by must be 'tag' or 'source'"):
        export_concept_map_markdown([], [], group_by=group_by)


@pytest.mark.parametrize("max_links_per_unit", [-1, 1.5, True])
def test_export_concept_map_validates_max_links_per_unit(max_links_per_unit):
    with pytest.raises(
        ValueError, match="max_links_per_unit must be a non-negative integer"
    ):
        export_concept_map_markdown([], [], max_links_per_unit=max_links_per_unit)
