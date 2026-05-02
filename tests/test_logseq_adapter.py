from __future__ import annotations

import pytest

from graph.adapters.logseq import LogseqAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject


LOGSEQ_SAMPLE = """
{:pages
 [{:block/name "Projects"
   :block/uuid #uuid "page-projects"
   :block/children
   [{:block/content "Build [[Graph Tool]] #Research"
     :block/uuid #uuid "block-alpha"
     :block/refs [{:block/name "Graph Tool"}]
     :block/tags [{:block/name "Research"}]
     :block/children
     [{:block/content "Nested note references ((block-target)) and [[Graph Tool]] #DeepDive"
       :block/uuid #uuid "block-beta"}]}]}
  {:block/name "Graph Tool"
   :block/uuid #uuid "page-graph-tool"
   :block/children
   [{:block/content "Target block"
     :block/uuid #uuid "block-target"}]}]}
"""


def test_logseq_edn_export_ingests_pages_nested_blocks_tags_refs_and_contains_edges(tmp_path):
    export = tmp_path / "logseq.edn"
    export.write_text(LOGSEQ_SAMPLE, encoding="utf-8")

    first = LogseqAdapter(file_path=str(export)).ingest()
    second = LogseqAdapter(file_path=str(export)).ingest()

    assert [unit.source_id for unit in first.units] == [
        "logseq:block:block-alpha",
        "logseq:block:block-beta",
        "logseq:block:block-target",
        "logseq:page:page-graph-tool",
        "logseq:page:page-projects",
    ]
    assert [unit.source_id for unit in second.units] == [unit.source_id for unit in first.units]

    projects = next(unit for unit in first.units if unit.source_id == "logseq:page:page-projects")
    assert projects.source_project == SourceProject.LOGSEQ
    assert projects.source_entity_type == "page"
    assert projects.content_type == ContentType.ARTIFACT
    assert projects.title == "Projects"
    assert "Build [[Graph Tool]] #Research" in projects.content
    assert projects.tags == ["graph tool", "research", "deepdive"]
    assert projects.metadata["page_name"] == "Projects"
    assert projects.metadata["uuid"] == "page-projects"

    alpha = next(unit for unit in first.units if unit.source_id == "logseq:block:block-alpha")
    assert alpha.title == "Build [[Graph Tool]] #Research"
    assert alpha.tags == ["Research", "graph tool"]
    assert alpha.metadata["refs"] == ["Graph Tool"]
    assert alpha.metadata["page_name"] == "Projects"
    assert alpha.metadata["page_source_id"] == "logseq:page:page-projects"
    assert alpha.metadata["parent_source_id"] == "logseq:page:page-projects"
    assert alpha.metadata["position"] == [1, 1]

    beta = next(unit for unit in first.units if unit.source_id == "logseq:block:block-beta")
    assert beta.metadata["parent_source_id"] == "logseq:block:block-alpha"
    assert beta.metadata["refs"] == ["Graph Tool", "block-target"]

    assert [(edge.from_unit_id, edge.to_unit_id, edge.relation, edge.source) for edge in first.edges] == [
        (
            "logseq:block:block-alpha",
            "logseq:block:block-beta",
            EdgeRelation.CONTAINS,
            EdgeSource.SOURCE,
        ),
        (
            "logseq:page:page-graph-tool",
            "logseq:block:block-target",
            EdgeRelation.CONTAINS,
            EdgeSource.SOURCE,
        ),
        (
            "logseq:page:page-projects",
            "logseq:block:block-alpha",
            EdgeRelation.CONTAINS,
            EdgeSource.SOURCE,
        ),
    ]
    assert [edge.id for edge in first.edges] == [edge.id for edge in second.edges]


def test_logseq_respects_entity_types(tmp_path):
    export = tmp_path / "logseq.edn"
    export.write_text(LOGSEQ_SAMPLE, encoding="utf-8")

    pages = LogseqAdapter(file_path=str(export)).ingest(entity_types=["page"])
    blocks = LogseqAdapter(file_path=str(export)).ingest(entity_types=["block"])

    assert [unit.source_entity_type for unit in pages.units] == ["page", "page"]
    assert pages.edges == []
    assert [unit.source_entity_type for unit in blocks.units] == ["block", "block", "block"]
    assert [(edge.from_unit_id, edge.to_unit_id) for edge in blocks.edges] == [
        ("logseq:block:block-alpha", "logseq:block:block-beta")
    ]


def test_logseq_missing_and_malformed_paths_raise_clear_exceptions(tmp_path):
    with pytest.raises(FileNotFoundError, match="Logseq EDN export path does not exist"):
        LogseqAdapter(file_path=str(tmp_path / "missing.edn")).ingest()

    malformed = tmp_path / "bad.edn"
    malformed.write_text("{:pages [", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed Logseq EDN export"):
        LogseqAdapter(file_path=str(malformed)).ingest()


def test_logseq_adapter_is_registered():
    assert "logseq" in list_adapters()
    adapter = get_adapter("logseq", file_path="/tmp/logseq.edn")
    assert isinstance(adapter, LogseqAdapter)
    assert adapter.name == "logseq"
