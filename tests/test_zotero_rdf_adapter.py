"""Tests for Zotero RDF ingestion."""

from __future__ import annotations

import pytest

from graph.adapters.registry import get_adapter
from graph.adapters.zotero_rdf import ZoteroRdfAdapter
from graph.types.enums import SourceProject


ZOTERO_RDF = """<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:bib="http://purl.org/net/biblio#"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:dcterms="http://purl.org/dc/terms/"
    xmlns:foaf="http://xmlns.com/foaf/0.1/"
    xmlns:z="http://www.zotero.org/namespaces/export#">
  <bib:Article rdf:about="http://zotero.org/users/local/items/ABCD1234">
    <z:itemType>journalArticle</z:itemType>
    <z:itemKey>ABCD1234</z:itemKey>
    <dc:title>Graph Memory for Research Notes</dc:title>
    <bib:authors>
      <rdf:Seq>
        <rdf:li>
          <foaf:Person>
            <foaf:givenName>Ada</foaf:givenName>
            <foaf:surname>Lovelace</foaf:surname>
          </foaf:Person>
        </rdf:li>
        <rdf:li>
          <foaf:Person>
            <foaf:name>Grace Hopper</foaf:name>
          </foaf:Person>
        </rdf:li>
      </rdf:Seq>
    </bib:authors>
    <dcterms:date>2025-04-10</dcterms:date>
    <bib:journal>Journal of Local Knowledge</bib:journal>
    <dc:identifier>doi:10.1234/example.doi</dc:identifier>
    <dc:subject>knowledge graphs</dc:subject>
    <dc:subject>notes</dc:subject>
    <dcterms:abstract>Shows how local RDF exports can preserve citation metadata.</dcterms:abstract>
  </bib:Article>
  <bib:Book rdf:about="urn:zotero:item:book-without-optionals">
    <z:itemType>book</z:itemType>
    <dc:title>Plain Bibliography Book</dc:title>
    <bib:authors>
      <rdf:Seq>
        <rdf:li>
          <foaf:Person>
            <foaf:surname>Noether</foaf:surname>
          </foaf:Person>
        </rdf:li>
      </rdf:Seq>
    </bib:authors>
  </bib:Book>
</rdf:RDF>
"""


def test_zotero_rdf_ingests_bibliographic_items(tmp_path):
    path = tmp_path / "library.rdf"
    path.write_text(ZOTERO_RDF, encoding="utf-8")

    result = ZoteroRdfAdapter(path=str(path)).ingest()

    assert len(result.units) == 2
    assert result.edges == []

    by_title = {unit.title: unit for unit in result.units}

    article = by_title["Graph Memory for Research Notes"]
    assert article.source_project == SourceProject.ZOTERO_RDF
    assert article.source_id == "zotero:ABCD1234"
    assert article.source_entity_type == "zotero_rdf_item"
    assert article.title == "Graph Memory for Research Notes"
    assert article.metadata == {
        "zotero_key": "ABCD1234",
        "item_type": "journalArticle",
        "title": "Graph Memory for Research Notes",
        "creators": ["Lovelace, Ada", "Grace Hopper"],
        "publication_title": "Journal of Local Knowledge",
        "date": "2025-04-10",
        "doi": "10.1234/example.doi",
        "url": "",
        "abstract": "Shows how local RDF exports can preserve citation metadata.",
        "source_file": "library.rdf",
    }
    assert article.tags == ["knowledge graphs", "notes", "journalArticle"]
    assert "Creators: Lovelace, Ada; Grace Hopper" in article.content
    assert "Publication: Journal of Local Knowledge" in article.content
    assert "DOI: 10.1234/example.doi" in article.content
    assert article.created_at.isoformat() == "2025-04-10T00:00:00+00:00"
    assert article.updated_at.isoformat() == "2025-04-10T00:00:00+00:00"

    book = by_title["Plain Bibliography Book"]
    assert book.title == "Plain Bibliography Book"
    assert book.metadata["item_type"] == "book"
    assert book.metadata["creators"] == ["Noether"]
    assert book.metadata["date"] == ""
    assert book.metadata["doi"] == ""
    assert book.metadata["url"] == ""
    assert book.metadata["abstract"] == ""
    assert book.tags == ["book"]
    assert book.source_id.startswith("zotero_rdf:")


def test_zotero_rdf_entity_type_filter(tmp_path):
    path = tmp_path / "library.rdf"
    path.write_text(ZOTERO_RDF, encoding="utf-8")

    result = ZoteroRdfAdapter(path=str(path)).ingest(entity_types=["other"])

    assert result.units == []
    assert result.edges == []


def test_zotero_rdf_malformed_xml_raises_value_error(tmp_path):
    path = tmp_path / "bad.rdf"
    path.write_text("<rdf:RDF><bib:Article></rdf:RDF>", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed Zotero RDF/XML"):
        ZoteroRdfAdapter(path=str(path)).ingest()


def test_zotero_rdf_registered():
    adapter = get_adapter("zotero_rdf", path="/tmp/library.rdf")

    assert adapter.name == "zotero_rdf"
