from graph.rag.evidence_license_signal import analyze_evidence_license_signal


def test_classifies_common_open_and_permissive_licenses():
    report = analyze_evidence_license_signal(
        [
            {"id": "a", "metadata": {"license": "CC-BY 4.0"}},
            {"id": "b", "content": "Released under the MIT License."},
            {"id": "c", "content": "No license listed."},
        ]
    )
    assert [row["license"] for row in report["results"]] == ["CC-BY", "MIT", "unknown"]
    assert report["license_counts"] == {"open": 1, "permissive": 1, "unknown": 1}


def test_separates_restrictive_language():
    report = analyze_evidence_license_signal([{"id": "x", "content": "Copyright 2024. All rights reserved."}])
    assert report["results"][0]["classification"] == "restrictive"
