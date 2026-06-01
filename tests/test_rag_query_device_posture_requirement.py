from __future__ import annotations

from graph.rag.query_device_posture_requirement import detect_query_device_posture_requirements


def test_detects_device_posture_categories():
    rows = detect_query_device_posture_requirements(
        "Require managed devices, compliant device posture, disk encryption, OS version, "
        "jailbreak/root detection, and EDR or MDM."
    )

    assert [row["category"] for row in rows] == [
        "compliant_device",
        "disk_encryption",
        "edr_mdm",
        "jailbreak_root",
        "managed_device",
        "os_version",
    ]


def test_handles_acronyms_and_phrase_variants():
    rows = detect_query_device_posture_requirements("Need MDM enrollment, BitLocker, and minimum OS patch level.")

    assert [row["category"] for row in rows] == ["disk_encryption", "edr_mdm", "os_version"]


def test_unrelated_device_mentions_return_empty_list():
    assert detect_query_device_posture_requirements("Compare device screen sizes in the examples.") == []
