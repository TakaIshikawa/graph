from graph.rag import detect_query_pci_dss_requirement


def test_pci_dss_requirement_detects_card_compliance_categories():
    report = detect_query_pci_dss_requirement(
        "PCI DSS scope for cardholder data environment, PAN storage, SAQ A, merchant level 2, and tokenized card storage."
    )

    assert report["requires_pci_dss"] is True
    assert report["categories"] == [
        "cardholder_data_environment",
        "merchant_level",
        "pan_card_storage",
        "pci_dss",
        "saq",
        "tokenization_scope",
    ]
    assert report["matches"][0]["matched_text"] == "PCI DSS"
    assert {"matched_text", "category", "severity", "span"} <= report["matches"][0].keys()


def test_pci_dss_requirement_ignores_generic_payment_words():
    report = detect_query_pci_dss_requirement("Compare business credit options and payment processor pricing.")

    assert report["requires_pci_dss"] is False
    assert report["categories"] == []
    assert report["matches"] == []
