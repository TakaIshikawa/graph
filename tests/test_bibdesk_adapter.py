"""Tests for the BibDesk plist adapter."""

from __future__ import annotations

import plistlib

from graph.adapters.bibdesk import BibDeskAdapter
from graph.adapters.registry import get_adapter


def test_get_adapter_returns_bibdesk_instance():
    adapter = get_adapter("bibdesk", path="/tmp/library.plist")

    assert isinstance(adapter, BibDeskAdapter)
    assert adapter.name == "bibdesk"


def test_ingest_bibdesk_plist_publications_with_metadata(tmp_path):
    plist = tmp_path / "library.plist"
    fixture = {
        "publications": [
            {
                "Cite Key": "smith2024graph",
                "Title": "Semantic Personal Graphs",
                "Author": "Smith, Ada and Doe, Grace",
                "Year": "2024",
                "Journal": "Journal of Knowledge Systems",
                "Abstract": "A study of personal semantic graphs.",
                "DOI": "10.1000/example",
                "URL": "https://example.com/paper",
                "BibTeX Type": "article",
            },
            {
                "citeKey": "lee2025agents",
                "title": "Agent Evaluation",
                "authors": ["Lee, Robin"],
                "year": 2025,
                "booktitle": "Proceedings of AgentConf",
            },
        ]
    }
    with plist.open("wb") as handle:
        plistlib.dump(fixture, handle)

    result = BibDeskAdapter(path=str(plist)).ingest()

    assert [unit.source_id for unit in result.units] == [
        "library.plist:smith2024graph",
        "library.plist:lee2025agents",
    ]
    first = result.units[0]
    assert first.source_project == "bibdesk"
    assert first.source_entity_type == "publication"
    assert first.title == "Semantic Personal Graphs"
    assert "Title: Semantic Personal Graphs" in first.content
    assert "Authors: Smith, Ada; Doe, Grace" in first.content
    assert "Abstract: A study of personal semantic graphs." in first.content
    assert "Venue: Journal of Knowledge Systems" in first.content
    assert "Year: 2024" in first.content
    assert "DOI: 10.1000/example" in first.content
    assert "URL: https://example.com/paper" in first.content
    assert first.metadata == {
        "cite_key": "smith2024graph",
        "doi": "10.1000/example",
        "url": "https://example.com/paper",
        "publication_type": "article",
        "authors": ["Smith, Ada", "Doe, Grace"],
        "year": "2024",
        "source_file": "library.plist",
        "raw_keys": [
            "Abstract",
            "Author",
            "BibTeX Type",
            "Cite Key",
            "DOI",
            "Journal",
            "Title",
            "URL",
            "Year",
        ],
    }
    assert result.units[1].metadata["authors"] == ["Lee, Robin"]
    assert result.units[1].metadata["year"] == "2025"
    assert result.units[1].metadata["doi"] == ""
    assert result.units[1].metadata["url"] == ""
    assert result.edges == []


def test_ingest_top_level_list_and_missing_optional_fields(tmp_path):
    plist = tmp_path / "minimal.plist"
    with plist.open("wb") as handle:
        plistlib.dump([{"Title": "Minimal Publication"}], handle)

    result = BibDeskAdapter(path=str(plist)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Minimal Publication"
    assert unit.content == "Title: Minimal Publication"
    assert unit.metadata["cite_key"] == ""
    assert unit.metadata["authors"] == []
    assert unit.metadata["raw_keys"] == ["Title"]
