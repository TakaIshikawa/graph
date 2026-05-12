from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.duolingo_progress_csv import DuolingoProgressCsvAdapter
from graph.types.models import SyncState


def test_duolingo_progress_csv_ingests_lessons_and_practice(tmp_path):
    export = tmp_path / "duolingo.csv"
    export.write_text(
        "Timestamp,Language,Course,Skill,Lesson Type,XP,Crowns,Score,Streak Day,Mistakes\n"
        "2026-05-01T08:00:00Z,Spanish,Spanish from English,Basics,Lesson,15,2,98,10,1\n"
        "2026-05-02T08:00:00Z,French,French from English,Food,Practice,10,3,95,11,2\n",
        encoding="utf-8",
    )

    result = DuolingoProgressCsvAdapter(path=str(export)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["lesson", "practice"]
    lesson = result.units[0]
    practice = result.units[1]
    assert lesson.metadata["xp"] == 15
    assert lesson.metadata["crowns"] == 2
    assert lesson.metadata["score"] == 98.0
    assert lesson.metadata["streak_day"] == 10
    assert lesson.metadata["mistakes"] == 1
    assert lesson.tags == ["duolingo", "spanish", "lesson"]
    assert practice.tags == ["duolingo", "french", "practice"]
    assert "Skill: Basics" in lesson.content


def test_duolingo_progress_csv_handles_optional_columns_and_skips_missing_context(tmp_path):
    export = tmp_path / "duolingo.csv"
    export.write_text(
        "Date,Time,Language,Skill,XP\n"
        "2026-05-01,08:00,Japanese,Hiragana,20\n"
        "2026-05-02,08:00,,Katakana,20\n"
        "2026-05-03,,German,Basics,5\n",
        encoding="utf-8",
    )

    result = DuolingoProgressCsvAdapter(path=str(export)).ingest()

    assert [unit.metadata["language"] for unit in result.units] == ["Japanese", "German"]
    assert "score" not in result.units[0].metadata


def test_duolingo_progress_csv_filters_since_and_entity_types(tmp_path):
    export = tmp_path / "duolingo.csv"
    export.write_text(
        "Timestamp,Language,Skill,Lesson Type,XP\n"
        "2026-05-01T08:00:00Z,Spanish,Basics,Lesson,15\n"
        "2026-05-03T08:00:00Z,Spanish,Basics,Practice,10\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="duolingo_progress_csv", source_entity_type="lesson", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = DuolingoProgressCsvAdapter(path=str(export)).ingest(since=since)
    lessons = DuolingoProgressCsvAdapter(path=str(export)).ingest(entity_types=["lesson"])

    assert [unit.source_entity_type for unit in result.units] == ["practice"]
    assert [unit.source_entity_type for unit in lessons.units] == ["lesson"]


def test_duolingo_progress_csv_course_and_skill_aggregates(tmp_path):
    export = tmp_path / "duolingo.csv"
    export.write_text(
        "Timestamp,Language,Course,Skill,Lesson Type,XP,Score,Mistakes\n"
        "2026-05-01T08:00:00Z,Spanish,Spanish from English,Basics,Lesson,15,90,1\n"
        "2026-05-02T08:00:00Z,Spanish,Spanish from English,Basics,Practice,10,100,2\n",
        encoding="utf-8",
    )

    result = DuolingoProgressCsvAdapter(path=str(export)).ingest(entity_types=["course", "skill", "lesson", "practice"])
    course = next(unit for unit in result.units if unit.source_entity_type == "course")
    skill = next(unit for unit in result.units if unit.source_entity_type == "skill")

    assert course.metadata["lesson_count"] == 1
    assert course.metadata["practice_count"] == 1
    assert course.metadata["total_xp"] == 25
    assert course.metadata["average_score"] == 95.0
    assert skill.metadata["mistake_total"] == 3
    assert {edge.metadata["relation_type"] for edge in result.edges} == {"course_contains_skill", "skill_contains_activity"}
