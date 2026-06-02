from __future__ import annotations

from graph.adapters.miro_boards_csv import MiroBoardsCsvAdapter
from graph.adapters.registry import get_adapter


def test_miro_boards_csv_ingests_boards(tmp_path):
    export = tmp_path / "miro.csv"
    export.write_text("Board ID,Name,Description,Owner,Team,Board URL,Created At,Modified At,Last Opened At,Access Level\nb1,Roadmap,Planning,ada,Product,https://miro.com/b1,2026-05-01T00:00:00Z,2026-05-02T00:00:00Z,2026-05-03T00:00:00Z,team\n", encoding="utf-8")

    unit = MiroBoardsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_entity_type == "board"
    assert unit.metadata["board_id"] == "b1"
    assert unit.metadata["access_level"] == "team"
    assert "https://miro.com/b1" in unit.content


def test_miro_boards_csv_searchable_without_description(tmp_path):
    export = tmp_path / "miro.csv"
    export.write_text("Name,Board URL\nRoadmap,https://miro.com/b1\n", encoding="utf-8")

    unit = MiroBoardsCsvAdapter(path=str(export)).ingest().units[0]
    assert "Roadmap" in unit.content
    assert "https://miro.com/b1" in unit.content


def test_miro_boards_csv_is_registered():
    assert isinstance(get_adapter("miro-boards-csv"), MiroBoardsCsvAdapter)
