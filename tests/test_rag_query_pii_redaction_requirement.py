from graph.rag.query_pii_redaction_requirement import detect_pii_redaction_requirements


def test_detects_pii_redaction_masking_and_privacy_transforms():
    rows = detect_pii_redaction_requirements(
        "Redact PII before retrieval, mask personal data in exports, anonymize profiles, "
        "pseudonymize users, and tokenization for identifiers."
    )

    assert rows == [
        {"cue": "Redact PII", "category": "redaction", "context": "Redact PII before retrieval, mask personal dat"},
        {
            "cue": "mask personal data",
            "category": "masking",
            "context": "Redact PII before retrieval, mask personal data in exports, anonymize profiles, pse",
        },
        {
            "cue": "anonymize",
            "category": "anonymization",
            "context": "val, mask personal data in exports, anonymize profiles, pseudonymize users, and t",
        },
        {
            "cue": "pseudonymize",
            "category": "pseudonymization",
            "context": "ata in exports, anonymize profiles, pseudonymize users, and tokenization for identif",
        },
        {
            "cue": "tokenization",
            "category": "tokenization",
            "context": "e profiles, pseudonymize users, and tokenization for identifiers.",
        },
    ]


def test_detects_log_redaction_sensitive_fields_and_data_minimization():
    rows = detect_pii_redaction_requirements(
        "Require logs redaction for sensitive fields and data minimization for customer records."
    )

    assert rows == [
        {
            "cue": "sensitive fields",
            "category": "sensitive_fields",
            "context": "Require logs redaction for sensitive fields and data minimization for customer",
        },
        {
            "cue": "logs redaction",
            "category": "log_redaction",
            "context": "Require logs redaction for sensitive fields and data minim",
        },
        {
            "cue": "data minimization",
            "category": "data_minimization",
            "context": "redaction for sensitive fields and data minimization for customer records.",
        },
    ]


def test_generic_privacy_wording_without_redaction_intent_has_no_match():
    assert detect_pii_redaction_requirements(
        "Explain privacy notices for personal data collection and consent."
    ) == []


def test_repeated_and_overlapping_cues_are_deterministic():
    query = "Redact PII, redact personal data, mask sensitive fields, and sensitive field masking."

    expected = [
        {
            "cue": "Redact PII",
            "category": "redaction",
            "context": "Redact PII, redact personal data, mask sensiti",
        },
        {
            "cue": "mask sensitive fields",
            "category": "masking",
            "context": "Redact PII, redact personal data, mask sensitive fields, and sensitive field masking.",
        },
        {
            "cue": "sensitive fields",
            "category": "sensitive_fields",
            "context": "act PII, redact personal data, mask sensitive fields, and sensitive field masking.",
        },
    ]
    assert detect_pii_redaction_requirements(query) == expected
    assert detect_pii_redaction_requirements(query) == expected
