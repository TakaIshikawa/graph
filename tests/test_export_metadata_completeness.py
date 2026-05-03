from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.export import export_metadata_completeness_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    title: str | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title if title is not None else f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
        metadata=metadata or {},
    )


def test_metadata_completeness_writes_summary_coverage_and_examples(tmp_path):
    path = tmp_path / "reports" / "metadata.md"

    stats = export_metadata_completeness_markdown(
        [
            unit("unit-b", metadata={"author": "A. Smith", "doi": "", "url": "https://b"}),
            unit("unit-a", metadata={"author": "R. Chen", "doi": "10.123/a"}),
            unit("unit-c", metadata={"author": [], "doi": None, "url": "https://c"}),
        ],
        path,
        required_keys=["author", "doi"],
        optional_keys=["url"],
    )

    report = path.read_text(encoding="utf-8")
    assert stats == {
        "path": str(path),
        "units_scanned": 3,
        "required_keys": ["author", "doi"],
        "optional_keys": ["url"],
        "missing_required_counts": {"author": 1, "doi": 2},
    }
    assert "| Units scanned | 3 |" in report
    assert "| Units missing required keys | 2 |" in report
    assert "| author | 2 | 1 | 66.7% |" in report
    assert "| doi | 1 | 2 | 33.3% |" in report
    assert "| url | 2 | 1 | 66.7% |" in report
    assert "| author | 1 | unit-c (Title unit-c) |" in report
    assert "| doi | 2 | unit-b (Title unit-b); unit-c (Title unit-c) |" in report


def test_metadata_completeness_treats_empty_values_as_absent(tmp_path):
    path = tmp_path / "metadata.md"

    export_metadata_completeness_markdown(
        [
            unit(
                "unit-a",
                metadata={
                    "none": None,
                    "empty_string": " \n ",
                    "empty_list": [],
                    "empty_dict": {},
                    "empty_tuple_is_present": (),
                    "false_is_present": False,
                },
            )
        ],
        path,
        required_keys=[
            "none",
            "empty_string",
            "empty_list",
            "empty_dict",
            "empty_tuple_is_present",
            "false_is_present",
        ],
    )

    report = path.read_text(encoding="utf-8")
    assert "| none | 0 | 1 | 0.0% |" in report
    assert "| empty_string | 0 | 1 | 0.0% |" in report
    assert "| empty_list | 0 | 1 | 0.0% |" in report
    assert "| empty_dict | 0 | 1 | 0.0% |" in report
    assert "| empty_tuple_is_present | 1 | 0 | 100.0% |" in report
    assert "| false_is_present | 1 | 0 | 100.0% |" in report


def test_metadata_completeness_is_deterministic_for_reordered_units_and_keys(tmp_path):
    units = [
        unit("unit-b", title="Beta | Title", metadata={"zeta": "yes", "alpha": ""}),
        unit("unit-a", title="Alpha\nTitle", metadata={"alpha": "yes"}),
    ]

    first_path = tmp_path / "first.md"
    second_path = tmp_path / "second.md"
    first_stats = export_metadata_completeness_markdown(
        units,
        first_path,
        required_keys=["zeta", "alpha"],
    )
    second_stats = export_metadata_completeness_markdown(
        reversed(units),
        second_path,
        required_keys=["alpha", "zeta"],
    )

    first_report = first_path.read_text(encoding="utf-8")
    second_report = second_path.read_text(encoding="utf-8")
    assert first_report == second_report
    assert first_stats["missing_required_counts"] == second_stats["missing_required_counts"]
    assert "| alpha | 1 | 1 | 50.0% |" in first_report
    assert "| zeta | 1 | 1 | 50.0% |" in first_report
    assert "| alpha | 1 | unit-b (Beta \\| Title) |" in first_report
    assert "| zeta | 1 | unit-a (Alpha Title) |" in first_report


def test_metadata_completeness_infers_optional_keys_and_escapes_cells(tmp_path):
    path = tmp_path / "metadata.md"

    export_metadata_completeness_markdown(
        [
            unit("unit-a", metadata={"source|url": "https://example.com", "required": "yes"}),
            unit("unit-b", metadata={"source|url": ""}),
        ],
        path,
        required_keys=["required"],
    )

    report = path.read_text(encoding="utf-8")
    assert "| source\\|url | 1 | 1 | 50.0% |" in report
    assert "| required | 1 | 1 | 50.0% |" in report


def test_metadata_completeness_respects_max_examples(tmp_path):
    path = tmp_path / "metadata.md"

    export_metadata_completeness_markdown(
        [unit("unit-b"), unit("unit-a"), unit("unit-c")],
        path,
        required_keys=["doi"],
        max_examples=2,
    )

    report = path.read_text(encoding="utf-8")
    assert "| doi | 3 | unit-a (Title unit-a); unit-b (Title unit-b) |" in report
    assert "unit-c" not in report


def test_metadata_completeness_rejects_invalid_max_examples(tmp_path):
    with pytest.raises(ValueError, match="max_examples must be a non-negative integer"):
        export_metadata_completeness_markdown([], tmp_path / "metadata.md", max_examples=-1)

    with pytest.raises(ValueError, match="max_examples must be a non-negative integer"):
        export_metadata_completeness_markdown([], tmp_path / "metadata.md", max_examples=True)


def test_metadata_completeness_is_importable_from_graph_export():
    from graph.export import export_metadata_completeness_markdown as imported

    assert imported is export_metadata_completeness_markdown
