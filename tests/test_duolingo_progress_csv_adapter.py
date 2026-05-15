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


def test_duolingo_progress_csv_language_aggregates_and_course_edges(tmp_path):
    export = tmp_path / "duolingo.csv"
    export.write_text(
        "Timestamp,Language,Course,Skill,Lesson Type,XP,Score,Mistakes\n"
        "2026-05-01T08:00:00Z,Spanish,Spanish from English,Basics,Lesson,15,90,1\n"
        "2026-05-02T08:00:00Z,spanish,Spanish from English,Food,Practice,10,100,2\n"
        "2026-05-03T08:00:00Z,French,French from English,Food,Lesson,20,80,0\n",
        encoding="utf-8",
    )

    result = DuolingoProgressCsvAdapter(path=str(export)).ingest(entity_types=["language", "course", "skill", "lesson", "practice"])

    languages = [unit for unit in result.units if unit.source_entity_type == "language"]
    courses = [unit for unit in result.units if unit.source_entity_type == "course"]
    skills = [unit for unit in result.units if unit.source_entity_type == "skill"]
    activities = [unit for unit in result.units if unit.source_entity_type in {"lesson", "practice"}]
    assert len(languages) == 2

    spanish = next(unit for unit in languages if unit.metadata["normalized_language"] == "spanish")
    spanish_activities = [unit for unit in activities if str(unit.metadata.get("language", "")).casefold() == "spanish"]
    assert spanish.source_id.startswith("duolingo_progress_csv:language:")
    assert spanish.metadata["language"] == "Spanish"
    assert spanish.metadata["course_count"] == 1
    assert spanish.metadata["skill_count"] == 2
    assert spanish.metadata["lesson_count"] == 1
    assert spanish.metadata["practice_count"] == 1
    assert spanish.metadata["total_xp"] == 25
    assert spanish.metadata["average_score"] == 95.0
    assert spanish.metadata["first_completed_at"] == "2026-05-01T08:00:00+00:00"
    assert spanish.metadata["last_completed_at"] == "2026-05-02T08:00:00+00:00"
    assert spanish.metadata["course_source_ids"] == sorted(
        course.source_id for course in courses if str(course.metadata.get("language", "")).casefold() == "spanish"
    )
    assert spanish.metadata["skill_source_ids"] == sorted(
        skill.source_id for skill in skills if str(skill.metadata.get("language", "")).casefold() == "spanish"
    )
    assert spanish.metadata["activity_source_ids"] == [unit.source_id for unit in spanish_activities]

    assert any(
        edge.from_unit_id == spanish.source_id
        and edge.to_unit_id in spanish.metadata["course_source_ids"]
        and edge.metadata["relation_type"] == "language_contains_course"
        for edge in result.edges
    )
    assert "language_contains_skill" not in {edge.metadata["relation_type"] for edge in result.edges}


def test_duolingo_progress_csv_language_to_skill_edges_without_courses(tmp_path):
    export = tmp_path / "duolingo.csv"
    export.write_text(
        "Timestamp,Language,Skill,Lesson Type,XP\n"
        "2026-05-01T08:00:00Z,Spanish,Basics,Lesson,15\n",
        encoding="utf-8",
    )

    result = DuolingoProgressCsvAdapter(path=str(export)).ingest(entity_types=["language", "skill"])

    language = next(unit for unit in result.units if unit.source_entity_type == "language")
    skill = next(unit for unit in result.units if unit.source_entity_type == "skill")
    assert {(edge.from_unit_id, edge.to_unit_id, edge.metadata["relation_type"]) for edge in result.edges} == {
        (language.source_id, skill.source_id, "language_contains_skill")
    }


def test_duolingo_progress_csv_streak_aggregates_group_by_language_and_streak_day(tmp_path):
    export = tmp_path / "duolingo.csv"
    export.write_text(
        "Timestamp,Language,Skill,Lesson Type,XP,Score,Streak Day,Mistakes\n"
        "2026-05-01T08:00:00Z,Spanish,Basics,Lesson,15,90,10,1\n"
        "2026-05-01T09:30:00Z,spanish,Food,Practice,10,100,10,0\n"
        "2026-05-02T08:00:00Z,Spanish,Basics,Lesson,20,80,11,2\n"
        "2026-05-01T08:30:00Z,French,Food,Lesson,5,70,10,3\n",
        encoding="utf-8",
    )

    result = DuolingoProgressCsvAdapter(path=str(export)).ingest(entity_types=["streak"])

    assert "streak" in DuolingoProgressCsvAdapter(path=str(export)).entity_types
    streaks = [unit for unit in result.units if unit.source_entity_type == "streak"]
    assert [(unit.metadata["normalized_language"], unit.metadata["streak_identity_type"], unit.metadata["streak_identity"]) for unit in streaks] == [
        ("french", "streak_day", "10"),
        ("spanish", "streak_day", "10"),
        ("spanish", "streak_day", "11"),
    ]

    spanish_day_10 = next(unit for unit in streaks if unit.metadata["normalized_language"] == "spanish" and unit.metadata["streak_day"] == 10)
    assert spanish_day_10.source_id.startswith("duolingo_progress_csv:streak:")
    assert spanish_day_10.metadata["language"] == "Spanish"
    assert spanish_day_10.metadata["activity_count"] == 2
    assert spanish_day_10.metadata["lesson_count"] == 1
    assert spanish_day_10.metadata["practice_count"] == 1
    assert spanish_day_10.metadata["total_xp"] == 25
    assert spanish_day_10.metadata["average_score"] == 95.0
    assert spanish_day_10.metadata["mistake_total"] == 1
    assert spanish_day_10.metadata["skills"] == ["Basics", "Food"]
    assert spanish_day_10.metadata["first_completed_at"] == "2026-05-01T08:00:00+00:00"
    assert spanish_day_10.metadata["last_completed_at"] == "2026-05-01T09:30:00+00:00"
    assert len(spanish_day_10.metadata["activity_source_ids"]) == 2
    assert result.edges == []


def test_duolingo_progress_csv_streak_aggregates_fallback_to_completed_date_and_activity_edges(tmp_path):
    export = tmp_path / "duolingo.csv"
    export.write_text(
        "Timestamp,Language,Skill,Lesson Type,XP\n"
        "2026-05-01T08:00:00Z,Spanish,Basics,Lesson,15\n"
        "2026-05-01T09:30:00Z,Spanish,Food,Practice,10\n"
        "2026-05-02T08:00:00Z,Spanish,Basics,Lesson,20\n",
        encoding="utf-8",
    )

    result = DuolingoProgressCsvAdapter(path=str(export)).ingest(entity_types=["streak", "lesson", "practice"])

    streaks = [unit for unit in result.units if unit.source_entity_type == "streak"]
    activities = [unit for unit in result.units if unit.source_entity_type in {"lesson", "practice"}]
    may_1 = next(unit for unit in streaks if unit.metadata["completed_date"] == "2026-05-01")
    may_1_activities = [unit for unit in activities if unit.metadata["completed_at"].startswith("2026-05-01")]

    assert [(unit.metadata["streak_identity_type"], unit.metadata["streak_identity"]) for unit in streaks] == [
        ("completed_date", "2026-05-01"),
        ("completed_date", "2026-05-02"),
    ]
    assert may_1.metadata["activity_count"] == 2
    assert may_1.metadata["total_xp"] == 25
    assert may_1.metadata["skills"] == ["Basics", "Food"]
    assert may_1.metadata["activity_source_ids"] == [unit.source_id for unit in may_1_activities]
    assert {
        (edge.from_unit_id, edge.to_unit_id, edge.metadata["relation_type"])
        for edge in result.edges
        if edge.from_unit_id == may_1.source_id
    } == {(may_1.source_id, unit.source_id, "streak_contains_activity") for unit in may_1_activities}
