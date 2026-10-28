import numpy as np
import pandas as pd

from ids.data.clean import clean_frame
from ids.data.labels import LabelMapping


def test_cleaning_coerces_invalid_numbers_and_infinity_without_dropping_rows():
    frame = pd.DataFrame({" Flow Duration ": ["10", "bad", np.inf], "Label": ["BENIGN", "DoS Hulk", "BENIGN"]})
    cleaned, report = clean_frame(frame, source_file="x.csv", mapping=LabelMapping("BENIGN", {"BENIGN": "BENIGN", "DoS Hulk": "DoS"}))
    assert len(cleaned) == 3
    assert cleaned["Flow Duration"].isna().sum() == 2
    assert report["dropped_rows"] == 0
    assert report["infinite_values_replaced"]["Flow Duration"] == 1


def test_cleaning_flags_duplicates_and_preserves_group_metadata():
    frame = pd.DataFrame(
        {
            "Flow Duration": [10, 10],
            "capture_day": ["day1", "day1"],
            "Label": ["BENIGN", "BENIGN"],
        }
    )
    cleaned, report = clean_frame(frame, source_file="x.csv", mapping=LabelMapping("BENIGN", {"BENIGN": "BENIGN"}))
    assert report["exact_duplicate_rows"] == 1
    assert report["duplicate_flagged_rows"] == 2
    assert cleaned["duplicate_row"].tolist() == [1, 1]
    assert cleaned["capture_day"].tolist() == ["day1", "day1"]


def test_cleaning_can_downcast_features_and_maps_corrupted_web_attack_labels():
    from ids.data.labels import mapping_from_config
    from ids.utils.io import read_yaml

    mapping = mapping_from_config(read_yaml("configs/label_map.yaml"))
    frame = pd.DataFrame({"Flow Duration": [10, 20], "Label": ["BENIGN", "Web Attack � XSS"]})
    cleaned, _ = clean_frame(frame, source_file="x.csv", mapping=mapping, numeric_dtype="float32")
    assert cleaned["Flow Duration"].dtype == np.float32
    assert cleaned["multiclass_label"].tolist() == ["BENIGN", "WebAttack"]
