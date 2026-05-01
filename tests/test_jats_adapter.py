"""Tests for the JATS XML adapter."""

from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.jats import JatsAdapter
from graph.types.enums import ContentType, SourceProject


JATS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<article article-type="research-article" xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <journal-meta>
      <journal-title-group>
        <journal-title>Journal of Test Articles</journal-title>
      </journal-title-group>
      <publisher>
        <publisher-name>Example Publisher</publisher-name>
      </publisher>
    </journal-meta>
    <article-meta>
      <article-id pub-id-type="doi">10.1234/example.doi</article-id>
      <article-categories>
        <subj-group>
          <subject>Open Access</subject>
          <subject>Knowledge Graphs</subject>
        </subj-group>
      </article-categories>
      <title-group>
        <article-title>Minimal <italic>JATS</italic> Article</article-title>
      </title-group>
      <contrib-group>
        <contrib contrib-type="author">
          <name>
            <surname>Lovelace</surname>
            <given-names>Ada</given-names>
          </name>
        </contrib>
        <contrib contrib-type="author">
          <name>
            <surname>Hopper</surname>
            <given-names>Grace</given-names>
          </name>
        </contrib>
      </contrib-group>
      <pub-date date-type="pub">
        <year>2024</year>
        <month>05</month>
        <day>09</day>
      </pub-date>
      <abstract>
        <p>This article demonstrates JATS ingestion.</p>
      </abstract>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Introduction</title>
      <p>The body paragraph is preserved as article content.</p>
    </sec>
  </body>
</article>
"""


def test_ingest_minimal_jats_article(tmp_path):
    article = tmp_path / "article.xml"
    article.write_text(JATS_XML, encoding="utf-8")

    result = JatsAdapter(path=str(article)).ingest()

    assert result.edges == []
    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.JATS
    assert unit.source_id == "doi:10.1234/example.doi"
    assert unit.source_entity_type == "article"
    assert unit.title == "Minimal JATS Article"
    assert unit.content_type == ContentType.FINDING
    assert "This article demonstrates JATS ingestion." in unit.content
    assert "The body paragraph is preserved as article content." in unit.content
    assert unit.tags == ["Open Access", "Knowledge Graphs"]
    assert unit.created_at == datetime(2024, 5, 9, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2024, 5, 9, tzinfo=timezone.utc)
    assert unit.metadata == {
        "doi": "10.1234/example.doi",
        "journal_title": "Journal of Test Articles",
        "publisher_name": "Example Publisher",
        "authors": ["Ada Lovelace", "Grace Hopper"],
        "published_date_parts": {"year": 2024, "month": 5, "day": 9},
        "path": "article.xml",
        "article_type": "research-article",
    }


def test_ingest_directory_recursively_and_skips_bad_xml(tmp_path):
    root = tmp_path / "exports"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "article.xml").write_text(JATS_XML, encoding="utf-8")
    (root / "non_jats.xml").write_text("<root><title>Not JATS</title></root>", encoding="utf-8")
    (root / "malformed.xml").write_text("<article><front>", encoding="utf-8")
    (nested / "ignore.txt").write_text(JATS_XML, encoding="utf-8")

    result = JatsAdapter(path=str(root)).ingest()
    filtered = JatsAdapter(path=str(root)).ingest(entity_types=["book"])
    missing = JatsAdapter(path=str(tmp_path / "missing")).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["path"] == "nested/article.xml"
    assert filtered.units == []
    assert missing.units == []


def test_ingest_jats_article_without_doi_uses_path_hash(tmp_path):
    article = tmp_path / "article.xml"
    article.write_text(
        JATS_XML.replace(
            '<article-id pub-id-type="doi">10.1234/example.doi</article-id>',
            "",
        ),
        encoding="utf-8",
    )

    result = JatsAdapter(path=str(article)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_id.startswith("jats:")
    assert result.units[0].metadata["doi"] == ""
