from __future__ import annotations

from graph.adapters.kindle_vocabulary_csv import KindleVocabularyCsvAdapter
from graph.adapters.registry import get_adapter


def test_kindle_vocabulary_csv_ingests_case_insensitive_headers(tmp_path):
    path = tmp_path / "vocab.csv"
    path.write_text("word,book title,book author,usage,lookup,date added,mastery\nserendipity,Book,Ada,Found it.,good luck,2026-01-01,Learned\n", encoding="utf-8")

    unit = KindleVocabularyCsvAdapter(str(path)).ingest().units[0]

    assert unit.title == "serendipity - Book"
    assert "Context: Found it." in unit.content
    assert unit.metadata["lookup"] == "good luck"
    assert unit.metadata["mastery"] == "Learned"
    assert isinstance(get_adapter("kindle_vocabulary_csv"), KindleVocabularyCsvAdapter)
