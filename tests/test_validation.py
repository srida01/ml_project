import pandas as pd
import pytest

from ids.data.validate import feature_columns, overlap_report, validate_schema


def test_feature_columns_exclude_targets_and_metadata():
    frame = pd.DataFrame({"x": [1, 2], "Label": ["BENIGN", "Bot"], "binary_label": [0, 1], "source_file": ["a", "b"], "capture_day": [1, 2], "Src Port": [10, 20], "Timestamp": [1, 2]})
    assert feature_columns(frame) == ["x"]


def test_schema_reports_missing_columns():
    with pytest.raises(ValueError, match="Missing required"):
        validate_schema(pd.DataFrame({"x": [1]}), ["x", "y"])


def test_overlap_report_counts_exact_feature_overlap():
    train = pd.DataFrame({"x": [1, 2]})
    validation = pd.DataFrame({"x": [2, 3]})
    test = pd.DataFrame({"x": [4]})
    assert overlap_report(train, validation, test, ["x"])["train_validation_feature_overlap"] == 1
