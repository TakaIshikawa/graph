from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

from graph.rag import analyze_result_date_coverage
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    metadata: dict | None = None,
    created_at: datetime | str | None = datetime(2024, 1, 1, tzinfo=timezone.utc),
) -> KnowledgeUnit:
    kwargs = {}
    if isinstance(created_at, datetime):
        kwargs["created_at"] = created_at
        kwargs["updated_at"] = created_at
        kwargs["ingested_at"] = created_at
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        **kwargs,
    )


@dataclass
class UnitStub:
    source_project: str
    metadata: dict


@dataclass
class ResultStub:
    unit: UnitStub


def test_analyze_result_date_coverage_counts_dates_across_result_shapes():
    report = analyze_result_date_coverage(
        [
            {"source_project": "top", "created_at": "2024-02-20T10:30:00Z"},
            {
                "source_project": "metadata",
                "metadata": {"published_at": "2024-03-05"},
            },
            {
                "unit": {
                    "source_project": "unit",
                    "updated_at": datetime(2024, 1, 15, 12, tzinfo=timezone.utc),
                }
            },
            ResultStub(
                UnitStub(
                    source_project="unit-metadata",
                    metadata={"published_at": date(2024, 4, 1)},
                )
            ),
        ]
    )

    assert report == {
        "total_results": 4,
        "dated_results": 4,
        "missing_date_results": 0,
        "invalid_date_results": 0,
        "earliest_date": "2024-01-15",
        "latest_date": "2024-04-01",
        "coverage_ratio": 1.0,
        "per_source": {
            "metadata": {
                "total_results": 1,
                "dated_results": 1,
                "missing_date_results": 0,
                "invalid_date_results": 0,
            },
            "top": {
                "total_results": 1,
                "dated_results": 1,
                "missing_date_results": 0,
                "invalid_date_results": 0,
            },
            "unit": {
                "total_results": 1,
                "dated_results": 1,
                "missing_date_results": 0,
                "invalid_date_results": 0,
            },
            "unit-metadata": {
                "total_results": 1,
                "dated_results": 1,
                "missing_date_results": 0,
                "invalid_date_results": 0,
            },
        },
    }


def test_analyze_result_date_coverage_separates_missing_and_invalid_dates():
    report = analyze_result_date_coverage(
        [
            {"source_project": "max", "created_at": "not-a-date"},
            {"source_project": "max", "created_at": 12345},
            {"source_project": "readwise", "created_at": ""},
            {"source_project": "readwise", "metadata": {"updated_at": None}},
            {"source_project": "readwise"},
            {"source_project": "max", "created_at": "2024-05-01"},
        ]
    )

    assert report["total_results"] == 6
    assert report["dated_results"] == 1
    assert report["invalid_date_results"] == 2
    assert report["missing_date_results"] == 3
    assert report["coverage_ratio"] == 1 / 6
    assert report["per_source"] == {
        "max": {
            "total_results": 3,
            "dated_results": 1,
            "missing_date_results": 0,
            "invalid_date_results": 2,
        },
        "readwise": {
            "total_results": 3,
            "dated_results": 0,
            "missing_date_results": 3,
            "invalid_date_results": 0,
        },
    }


def test_analyze_result_date_coverage_groups_unknown_sources_and_enum_values():
    report = analyze_result_date_coverage(
        [
            {"created_at": "2024-01-01"},
            {"source_project": " ", "created_at": "2024-01-02"},
            {
                "unit": unit(
                    "enum-source",
                    source_project=SourceProject.PINBOARD,
                    created_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
                )
            },
        ]
    )

    assert report["per_source"] == {
        "pinboard": {
            "total_results": 1,
            "dated_results": 1,
            "missing_date_results": 0,
            "invalid_date_results": 0,
        },
        "unknown": {
            "total_results": 2,
            "dated_results": 2,
            "missing_date_results": 0,
            "invalid_date_results": 0,
        },
    }


def test_analyze_result_date_coverage_supports_custom_date_keys():
    report = analyze_result_date_coverage(
        [
            {"source_project": "max", "metadata": {"seen_at": "2024-06-10"}},
            {"source_project": "max", "created_at": "2020-01-01"},
            {"source_project": "max", "metadata": {"seen_at": "invalid"}},
        ],
        date_keys=("seen_at",),
    )

    assert report["dated_results"] == 1
    assert report["missing_date_results"] == 1
    assert report["invalid_date_results"] == 1
    assert report["earliest_date"] == "2024-06-10"
    assert report["latest_date"] == "2024-06-10"


def test_analyze_result_date_coverage_is_importable_from_graph_rag():
    assert callable(analyze_result_date_coverage)
