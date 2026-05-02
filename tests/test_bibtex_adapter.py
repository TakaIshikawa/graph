"""Tests for the BibTeX bibliography adapter."""

from __future__ import annotations

from graph.adapters.bibtex import BibtexAdapter
from graph.adapters.registry import get_adapter, list_adapters


def test_ingest_multiple_bibtex_entries_with_common_metadata(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(
        r"""@article{smith2024graph,
  title = {Semantic Personal Graphs},
  author = {Smith, Ada and Doe, Grace},
  year = {2024},
  journal = {Journal of Knowledge Systems},
  abstract = {A study of personal semantic graphs.},
  keywords = {graphs, knowledge, graphs},
  doi = {10.1000/example},
  url = {https://example.com/paper}
}

@inproceedings{lee2025agents,
  title = "Agent Evaluation",
  author = "Lee, Robin",
  booktitle = {Proceedings of AgentConf},
  year = 2025,
  keywords = {agents; evaluation}
}
""",
        encoding="utf-8",
    )

    result = BibtexAdapter(path=str(bib)).ingest()

    assert [unit.source_id for unit in result.units] == [
        "refs.bib:smith2024graph",
        "refs.bib:lee2025agents",
    ]
    first = result.units[0]
    assert first.source_project == "bibtex"
    assert first.source_entity_type == "bibtex_entry"
    assert first.title == "Semantic Personal Graphs"
    assert first.tags == ["graphs", "knowledge"]
    assert first.metadata == {
        "citation_key": "smith2024graph",
        "entry_type": "article",
        "title": "Semantic Personal Graphs",
        "authors": ["Smith, Ada", "Doe, Grace"],
        "year": "2024",
        "doi": "10.1000/example",
        "url": "https://example.com/paper",
        "journal": "Journal of Knowledge Systems",
        "booktitle": "",
        "abstract": "A study of personal semantic graphs.",
        "keywords": ["graphs", "knowledge"],
        "source_file": "refs.bib",
    }
    assert "Title: Semantic Personal Graphs" in first.content
    assert "Authors: Smith, Ada; Doe, Grace" in first.content
    assert "Year: 2024" in first.content
    assert "Venue: Journal of Knowledge Systems" in first.content
    assert "Abstract: A study of personal semantic graphs." in first.content
    assert "DOI: 10.1000/example" in first.content
    assert "URL: https://example.com/paper" in first.content
    assert "Keywords: graphs, knowledge" in first.content
    assert result.units[1].metadata["booktitle"] == "Proceedings of AgentConf"
    assert result.units[1].tags == ["agents", "evaluation"]
    assert result.edges == []


def test_bibtex_parser_handles_braced_quoted_and_multiline_values(tmp_path):
    bib = tmp_path / "library.bib"
    bib.write_text(
        r'''@book{knuth1984texbook,
  title = "{The} TeXbook",
  author = {Knuth, Donald E.},
  year = "1984",
  abstract = {A multiline
    abstract with {protected} words
    and escaped \& symbols.},
  url = "https://example.com/texbook?one=1,two=2"
}
''',
        encoding="utf-8",
    )

    [unit] = BibtexAdapter(path=str(bib)).ingest().units

    assert unit.source_id == "library.bib:knuth1984texbook"
    assert unit.title == "The TeXbook"
    assert unit.metadata["authors"] == ["Knuth, Donald E."]
    assert unit.metadata["year"] == "1984"
    assert unit.metadata["abstract"] == (
        "A multiline abstract with protected words and escaped & symbols."
    )
    assert unit.metadata["url"] == "https://example.com/texbook?one=1,two=2"


def test_bibtex_missing_optional_fields_uses_citation_key_fallback(tmp_path):
    bib = tmp_path / "minimal.bib"
    bib.write_text("@misc{minimal_key}\n", encoding="utf-8")

    [unit] = BibtexAdapter(path=str(bib)).ingest().units

    assert unit.title == "minimal_key"
    assert unit.content == "Title: minimal_key"
    assert unit.metadata["authors"] == []
    assert unit.metadata["year"] == ""
    assert unit.metadata["doi"] == ""
    assert unit.metadata["url"] == ""
    assert unit.metadata["keywords"] == []
    assert unit.tags == []


def test_bibtex_adapter_is_registered():
    assert "bibtex" in list_adapters()
    adapter = get_adapter("bibtex", path="/tmp/refs.bib")
    assert isinstance(adapter, BibtexAdapter)
    assert adapter.name == "bibtex"
