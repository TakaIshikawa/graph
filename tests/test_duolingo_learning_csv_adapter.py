from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.duolingo_learning_csv import DuolingoLearningCsvAdapter
from graph.types.models import SyncState


def test_duolingo_learning_csv_ingests_learning_sessions_and_numbers(tmp_path):
    export = tmp_path / "duolingo-learning.csv"
    export.write_text(
        "Date,Course,Skill,Lesson,XP,Correct,Mistakes,Time Spent,Notes\n"
        "2026-05-01T08:00:00Z,Spanish,Basics,Greetings,15,12,1,5 min,Good pace\n"
        "2026-05-02T08:00:00Z,,,,20,,,3 min,\n"
        "2026-05-03T08:00:00Z,,,,,,,\n",
        encoding="utf-8",
    )

    result = DuolingoLearningCsvAdapter(path=str(export)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["learning_session", "learning_session"]
    first = result.units[0]
    assert first.metadata["course"] == "Spanish"
    assert first.metadata["skill"] == "Basics"
    assert first.metadata["lesson"] == "Greetings"
    assert first.metadata["xp"] == 15
    assert first.metadata["correct"] == 12
    assert first.metadata["mistakes"] == 1
    assert "Time spent: 5 min" in first.content
    assert result.units[1].metadata["xp"] == 20


def test_duolingo_learning_csv_requires_date_plus_context_and_filters(tmp_path):
    export = tmp_path / "duolingo-learning.csv"
    export.write_text(
        "Date,Course,Skill,Lesson,XP\n"
        "2026-05-01T08:00:00Z,Spanish,Basics,Greetings,15\n"
        "2026-05-03T08:00:00Z,French,Food,Cafe,10\n"
        ",German,Basics,Intro,5\n",
        encoding="utf-8",
    )
    since = SyncState(
        source_project="duolingo_learning_csv",
        source_entity_type="learning_session",
        last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    result = DuolingoLearningCsvAdapter(path=str(export)).ingest(since=since)

    assert [unit.metadata["course"] for unit in result.units] == ["French"]
    assert DuolingoLearningCsvAdapter(path=str(export)).ingest(entity_types=["course"]).units == []
