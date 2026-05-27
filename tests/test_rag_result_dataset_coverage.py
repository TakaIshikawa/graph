from __future__ import annotations

from graph.rag.result_dataset_coverage import audit_result_dataset_coverage


def test_detects_dataset_bearing_results_from_text_urls_and_metadata():
    report = audit_result_dataset_coverage(
        [
            {"id": "text", "title": "Benchmark dataset for retrieval evaluation"},
            {"id": "url", "url": "https://zenodo.org/records/12345"},
            {
                "id": "metadata",
                "snippet": "Paper summary",
                "metadata": {"repository": "Hugging Face dataset card with supplementary data"},
            },
            {"id": "plain", "content": "A background editorial with no artifacts."},
        ]
    )

    assert report["has_dataset_coverage"] is True
    assert report["dataset_result_count"] == 3
    assert report["coverage_ratio"] == 0.75
    assert [source["result_id"] for source in report["dataset_sources"]] == ["text", "url", "metadata"]
    assert report["missing_dataset_result_ids"] == ["plain"]
    assert report["dataset_sources"][0]["dataset_cues"] == ["dataset", "benchmark"]
    assert report["dataset_sources"][1]["dataset_cues"] == ["zenodo"]
    assert report["dataset_sources"][2]["dataset_cues"] == [
        "dataset",
        "data_repository",
        "supplementary_data",
        "huggingface_dataset",
    ]


def test_detects_named_repositories_registries_and_github_releases():
    report = audit_result_dataset_coverage(
        [
            {"id": "figshare", "url": "https://figshare.com/articles/dataset/example/1"},
            {"id": "dryad", "content": "Archived in the Dryad data repository."},
            {"id": "kaggle", "metadata": {"source": "Kaggle competition dataset"}},
            {"id": "registry", "snippet": "Clinical registry export with a GitHub release."},
        ]
    )

    assert report["dataset_result_count"] == 4
    assert report["coverage_ratio"] == 1.0
    assert [source["dataset_cues"] for source in report["dataset_sources"]] == [
        ["dataset", "figshare"],
        ["dataset", "data_repository", "dryad"],
        ["dataset", "kaggle"],
        ["registry", "github_release"],
    ]
    assert report["missing_dataset_result_ids"] == []


def test_missing_dataset_result_ids_use_stable_fallbacks():
    report = audit_result_dataset_coverage([{"content": "General overview."}, {"id": "p2", "title": "News analysis"}])

    assert report == {
        "has_dataset_coverage": False,
        "dataset_result_count": 0,
        "coverage_ratio": 0.0,
        "dataset_sources": [],
        "missing_dataset_result_ids": ["result-1", "p2"],
    }


def test_empty_results_return_neutral_report():
    assert audit_result_dataset_coverage([]) == {
        "has_dataset_coverage": False,
        "dataset_result_count": 0,
        "coverage_ratio": 0.0,
        "dataset_sources": [],
        "missing_dataset_result_ids": [],
    }
