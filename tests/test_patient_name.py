"""Patient name helpers for clinical reports."""

from utils.report_utils import normalize_patient_name, patient_display_name


def test_normalize_patient_name():
    assert normalize_patient_name("  John   Smith  ") == "John Smith"


def test_patient_display_name_empty():
    assert patient_display_name("") == "Not Provided"
    assert patient_display_name("   ") == "Not Provided"


def test_patient_display_name_value():
    assert patient_display_name("Jane Doe") == "Jane Doe"
