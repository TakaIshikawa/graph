"""Tests for Crossref works JSON ingestion."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.crossref import CrossrefAdapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject


def _work(doi: str, title: str) -> dict:
    return {
        "DOI": doi,
        "title": [title],
        "abstract": "<jats:p>This paper studies graph ingestion.</jats:p>",
        "author": [
            {"given": "Ada", "family": "Lovelace"},
            {"name": "Graph Metadata Consortium"},
        ],
        "issued": {"date-parts": [[2025, 4, 3]]},
        "URL": f"https://doi.org/{doi}",
        "publisher": "Example Press",
        "container-title": ["Journal of Knowledge Graphs"],
        "subject": ["Computer Science", "Metadata"],
        "type": "journal-article",
    }


def test_ingests_crossref_message_items_with_normalized_metadata(tmp_path):
    path = tmp_path / "works.json"
    path.write_text(
        json.dumps({"message": {"items": [_work("10.5555/ABC.1", "Crossref Graphs")]}}),
        encoding="utf-8",
    )

    result = CrossrefAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    assert result.edges == []
    unit = result.units[0]
    assert unit.source_project == SourceProject.CROSSREF
    assert unit.source_entity_type == "work"
    assert unit.source_id == "doi:10.5555/abc.1"
    assert unit.title == "Crossref Graphs"
    assert unit.content_type == "finding"
    assert "Authors: Ada Lovelace; Graph Metadata Consortium" in unit.content
    assert "Abstract: This paper studies graph ingestion." in unit.content
    assert unit.metadata["doi"] == "10.5555/abc.1"
    assert unit.metadata["authors"] == ["Ada Lovelace", "Graph Metadata Consortium"]
    assert unit.metadata["publisher"] == "Example Press"
    assert unit.metadata["issued"] == "2025-04-03"
    assert unit.metadata["container_title"] == "Journal of Knowledge Graphs"
    assert unit.metadata["subjects"] == ["Computer Science", "Metadata"]
    assert unit.tags == ["Computer Science", "Metadata", "Journal of Knowledge Graphs"]
    assert unit.created_at == datetime(2025, 4, 3, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 4, 3, tzinfo=timezone.utc)


def test_ingests_direct_item_arrays_and_emits_reference_edges(tmp_path):
    cited = _work("10.5555/TARGET", "Referenced Work")
    citing = _work("https://doi.org/10.5555/SOURCE", "Citing Work")
    citing["reference"] = [
        {"DOI": "10.5555/TARGET"},
        {"doi": "https://doi.org/10.5555/missing"},
        {"DOI": "10.5555/TARGET"},
    ]
    path = tmp_path / "batch.json"
    path.write_text(json.dumps([citing, cited]), encoding="utf-8")

    result = CrossrefAdapter(path=str(path)).ingest()

    assert {unit.source_id for unit in result.units} == {
        "doi:10.5555/source",
        "doi:10.5555/target",
    }
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.from_unit_id == "doi:10.5555/source"
    assert edge.to_unit_id == "doi:10.5555/target"
    assert edge.relation == EdgeRelation.REFERENCES
    assert edge.source == EdgeSource.SOURCE
    assert edge.metadata["relation_type"] == "crossref_reference"
    assert edge.metadata["doi"] == "10.5555/target"


def test_ingests_json_files_from_directory(tmp_path):
    first = tmp_path / "first.json"
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "second.json"
    first.write_text(json.dumps({"message": {"items": [_work("10.5555/ONE", "One")]}}), encoding="utf-8")
    second.write_text(json.dumps([_work("10.5555/TWO", "Two")]), encoding="utf-8")

    result = CrossrefAdapter(path=str(tmp_path)).ingest()

    assert [unit.source_id for unit in result.units] == ["doi:10.5555/one", "doi:10.5555/two"]


def test_entity_filter_returns_empty_result(tmp_path):
    path = tmp_path / "works.json"
    path.write_text(json.dumps([_work("10.5555/ONE", "One")]), encoding="utf-8")

    result = CrossrefAdapter(path=str(path)).ingest(entity_types=["article"])

    assert result.units == []
    assert result.edges == []
