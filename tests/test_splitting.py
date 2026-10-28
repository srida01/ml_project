import pandas as pd

from ids.data.split import split_frame


def test_group_split_keeps_groups_disjoint():
    frame = pd.DataFrame({"binary_label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], "source_file": [f"d{i}.csv" for i in range(10)]})
    result = split_frame(frame, group_columns=["source_file"])
    assert set(result.manifest["groups"]["train"]).isdisjoint(result.manifest["groups"]["test"])
    assert len(result.train) + len(result.validation) + len(result.test) == len(frame)


def test_group_split_does_not_fall_back_with_five_groups():
    frame = pd.DataFrame(
        {
            "binary_label": [0, 1] * 5,
            "source_file": [f"d{i}.csv" for i in range(5) for _ in range(2)],
        }
    )
    result = split_frame(frame, group_columns=["source_file"])
    assert result.manifest["strategy"] == "group"
    assert result.manifest["groups"]["train"]
    assert result.manifest["groups"]["validation"]
    assert result.manifest["groups"]["test"]


def test_withheld_family_is_test_only():
    frame = pd.DataFrame(
        {
            "binary_label": [0, 1, 1, 0, 1, 1],
            "multiclass_label": ["BENIGN", "DoS", "Novel", "BENIGN", "DoS", "Novel"],
            "source_file": ["a", "a", "b", "c", "c", "b"],
        }
    )
    result = split_frame(frame, group_columns=["source_file"], withheld_label_values=["Novel"])
    assert "Novel" not in set(result.train["multiclass_label"])
    assert "Novel" not in set(result.validation["multiclass_label"])
    assert set(result.test["multiclass_label"]) >= {"Novel"}


def test_group_split_keeps_every_class_in_every_partition():
    # Mirrors CICIDS2017 day files: Monday is benign-only. With seed 42 the
    # first shuffle puts it in validation, leaving no attacks to calibrate on.
    days = ["friday", "monday", "thursday", "tuesday", "wednesday"]
    frame = pd.DataFrame(
        {
            "binary_label": [0 if day == "monday" else label for day in days for label in (0, 1)],
            "source_file": [f"{day}.csv" for day in days for _ in range(2)],
        }
    )
    result = split_frame(frame, group_columns=["source_file"], seed=42)
    assert result.manifest["strategy"] == "group"
    assert result.manifest["group_shuffle_attempts"] > 1
    for part in (result.train, result.validation, result.test):
        assert set(part["binary_label"]) == {0, 1}
