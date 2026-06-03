from __future__ import annotations

from graph.adapters.coursera_courses_csv import CourseraCoursesCsvAdapter


def test_coursera_courses_csv_emits_course_with_metadata(tmp_path):
    export = tmp_path / "courses.csv"
    export.write_text(
        "Course Title,Provider,Instructor,Completion Status,Progress %,Certificate URL,Completed Date,Course URL\n"
        "Machine Learning,Stanford,Andrew Ng,Completed,100,https://coursera.org/cert,2025-05-01,https://coursera.org/learn/ml\n",
        encoding="utf-8",
    )

    unit = CourseraCoursesCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_entity_type == "course"
    assert unit.metadata["course_title"] == "Machine Learning"
    assert unit.metadata["provider"] == "Stanford"
    assert unit.metadata["instructor"] == "Andrew Ng"
    assert unit.metadata["status"] == "Completed"
    assert unit.metadata["progress"] == 100.0
    assert unit.metadata["certificate_url"] == "https://coursera.org/cert"
    assert unit.metadata["course_url"] == "https://coursera.org/learn/ml"


def test_coursera_courses_csv_uses_title_provider_identity_and_handles_missing_optionals(tmp_path):
    export = tmp_path / "courses.csv"
    export.write_text("Title,University\nAlgorithms,Princeton\nAlgorithms,Stanford\n", encoding="utf-8")

    first = CourseraCoursesCsvAdapter(path=str(export)).ingest().units
    second = CourseraCoursesCsvAdapter(path=str(export)).ingest().units

    assert len(first) == 2
    assert first[0].metadata["course_title"] == "Algorithms"
    assert "progress" not in first[0].metadata
    assert [unit.source_id for unit in first] == [unit.source_id for unit in second]
