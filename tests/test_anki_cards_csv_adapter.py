from __future__ import annotations

from graph.adapters.anki_cards_csv import AnkiCardsCsvAdapter


def test_anki_cards_csv_supports_comma_exports_and_keeps_scheduling_metadata(tmp_path):
    export = tmp_path / "cards.csv"
    export.write_text(
        "Deck,Note Type,Front,Back,Tags,Due,Interval,Ease,Lapses\n"
        'Research,Basic,"What is a graph?","Nodes and edges","graph theory",2026-05-01,10,250,1\n',
        encoding="utf-8",
    )

    unit = AnkiCardsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "anki_cards_csv"
    assert unit.metadata["deck"] == "Research"
    assert unit.metadata["front"] == "What is a graph?"
    assert unit.metadata["back"] == "Nodes and edges"
    assert unit.metadata["tags"] == ["graph", "theory"]
    assert unit.metadata["interval"] == 10
    assert unit.metadata["ease"] == 250
    assert unit.metadata["lapses"] == 1
    assert "Front: What is a graph?" in unit.content
    assert "Back: Nodes and edges" in unit.content


def test_anki_cards_csv_supports_tab_delimited_exports(tmp_path):
    export = tmp_path / "cards.tsv"
    export.write_text("Deck\tFront\tBack\tTags\nJapanese\t猫\tcat\tlang vocab\n", encoding="utf-8")

    unit = AnkiCardsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.metadata["deck"] == "Japanese"
    assert unit.metadata["tags"] == ["lang", "vocab"]
