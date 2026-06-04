from graph.rag.query_dlp_requirement import detect_query_dlp_requirement


def test_detects_dlp_synonyms_and_structured_evidence_terms():
    report = detect_query_dlp_requirement(
        "Need DLP and data loss prevention evidence to prevent data exfiltration "
        "and block sensitive data leakage."
    )

    assert report["requires_dlp"] is True
    assert report["classification"] == "dlp_requirement"
    assert report["requirements"] == [
        {"category": "dlp", "matched_text": "DLP", "severity": "high", "evidence_terms": ["dlp"]},
        {
            "category": "exfiltration_prevention",
            "matched_text": "prevent data exfiltration",
            "severity": "high",
            "evidence_terms": ["prevent", "data", "exfiltration"],
        },
        {
            "category": "sensitive_data_leakage",
            "matched_text": "sensitive data leakage",
            "severity": "high",
            "evidence_terms": ["sensitive", "data", "leakage"],
        },
    ]
    assert {"dlp", "exfiltration", "leakage", "sensitive"} <= set(report["evidence_terms"])


def test_detects_clipboard_download_blocking_and_content_inspection():
    report = detect_query_dlp_requirement(
        "Require clipboard controls, restrict downloads, and inspect uploads for sensitive records."
    )

    assert report["requires_dlp"] is True
    assert [row["category"] for row in report["requirements"]] == [
        "clipboard_blocking",
        "content_inspection",
        "download_blocking",
    ]
    assert "clipboard" in report["evidence_terms"]
    assert "downloads" in report["evidence_terms"]
    assert "inspect" in report["evidence_terms"]


def test_broad_privacy_or_encryption_queries_do_not_trigger_dlp():
    assert detect_query_dlp_requirement("Summarize privacy requirements for customer data.") == {
        "requires_dlp": False,
        "classification": "unrelated",
        "requirements": [],
        "evidence_terms": [],
    }
    assert detect_query_dlp_requirement("Require encryption at rest and in transit for all records.") == {
        "requires_dlp": False,
        "classification": "unrelated",
        "requirements": [],
        "evidence_terms": [],
    }


def test_unrelated_compliance_queries_do_not_trigger_dlp():
    assert detect_query_dlp_requirement("Need SOC 2 audit evidence and GDPR lawful basis details.") == {
        "requires_dlp": False,
        "classification": "unrelated",
        "requirements": [],
        "evidence_terms": [],
    }
