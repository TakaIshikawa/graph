from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.coursera_progress_csv import CourseraProgressCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_coursera_progress_csv_ingests_progress_metadata_and_registry(tmp_path):
    export = tmp_path / "coursera.csv"
    export.write_text(
        "\n".join(
            [
                "course_title,specialization,lesson,module,progress,grade,status,started_at,completed_at,updated_at,certificate_url,course_url",
                "ML Course,AI Track,Intro,Week 1,85%,97.5,completed,2025-01-01,2025-01-02,2025-01-03T04:05:06Z,https://coursera.org/cert,https://coursera.org/learn/ml",
            ]
        ),
        encoding="utf-8",
    )

    result = CourseraProgressCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.COURSERA_PROGRESS_CSV
    assert unit.source_entity_type == "course_progress"
    assert unit.metadata["course"] == "ML Course"
    assert unit.metadata["specialization"] == "AI Track"
    assert unit.metadata["lesson"] == "Intro"
    assert unit.metadata["module"] == "Week 1"
    assert unit.metadata["progress"] == 85
    assert unit.metadata["grade"] == 97.5
    assert unit.metadata["status"] == "completed"
    assert unit.metadata["certificate_url"] == "https://coursera.org/cert"
    assert unit.metadata["course_url"] == "https://coursera.org/learn/ml"
    assert unit.metadata["source_file"] == "coursera.csv"
    assert unit.updated_at == datetime(2025, 1, 3, 4, 5, 6, tzinfo=timezone.utc)
    assert get_adapter("coursera_progress_csv", path=str(export)).name == "coursera_progress_csv"


def test_coursera_progress_csv_skips_empty_rows_invalid_numbers_since_and_filters(tmp_path):
    (tmp_path / "old.csv").write_text("course,item,progress,updated_at\nCourse,Old,10,2025-01-01\n", encoding="utf-8")
    (tmp_path / "new.csv").write_text("course,item,progress,grade,updated_at,course_url\nCourse,New,not-a-number,bad,2025-01-03,https://coursera.org/new\n,,,\n", encoding="utf-8")
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    adapter = CourseraProgressCsvAdapter(path=str(tmp_path))
    sync = SyncState(source_project="coursera_progress_csv", source_entity_type="course_progress", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["New"]
    assert "progress" not in first.units[0].metadata
    assert "grade" not in first.units[0].metadata
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["course"]).units == []
