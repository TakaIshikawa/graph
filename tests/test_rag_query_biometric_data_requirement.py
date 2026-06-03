from graph.rag import detect_query_biometric_data_requirement


def test_biometric_data_requirement_detects_modalities_and_privacy_controls():
    report = detect_query_biometric_data_requirement(
        "Biometric data program for face recognition, fingerprint biometrics, voiceprints, palm scans, liveness detection, biometric consent, biometric retention, and delete biometric data."
    )

    assert report["has_biometric_data_requirements"] is True
    assert report["categories"] == [
        "biometric_identifiers",
        "consent",
        "deletion",
        "face",
        "fingerprint",
        "liveness",
        "palm",
        "retention",
        "voice",
    ]
    assert report["matches"][0]["matched_text"] == "Biometric data"
    assert {"matched_text", "category", "severity", "span"} <= report["matches"][0].keys()


def test_biometric_data_requirement_ignores_generic_media_and_debugging():
    report = detect_query_biometric_data_requirement("Use generic photos, face-to-face meetings, fingerprints in debugging, and voice UI prompts.")

    assert report["has_biometric_data_requirements"] is False
    assert report["matches"] == []
