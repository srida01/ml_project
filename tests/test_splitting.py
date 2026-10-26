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
