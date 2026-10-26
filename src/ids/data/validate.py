"""Schema and leakage validations."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


DEFAULT_NON_FEATURE_COLUMNS = {
    "Label",
    "raw_label",
    "binary_label",
    "multiclass_label",
    "source_file",
    "source_row",
    "flow_id",
    "timestamp",
    "sequence_id",
    "split",
    "capture_day",
    "duplicate_row",
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Src Port",
    "Dst Port",
    "Protocol",
    "Timestamp",
}


def feature_columns(frame: pd.DataFrame, excluded: Iterable[str] = ()) -> list[str]:
    excluded_names = {str(column).strip().lower() for column in DEFAULT_NON_FEATURE_COLUMNS | set(excluded)}
    return [
        column
        for column in frame.columns
        if str(column).strip().lower() not in excluded_names and pd.api.types.is_numeric_dtype(frame[column])
    ]


def validate_schema(frame: pd.DataFrame, required: Iterable[str], allow_extra: bool = True) -> None:
    required = list(required)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not allow_extra:
        extra = [column for column in frame.columns if column not in required]
        if extra:
            raise ValueError(f"Unexpected columns: {extra}")


def assert_finite(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    columns = list(columns)
    bad = [column for column in columns if not pd.to_numeric(frame[column], errors="coerce").replace([float("inf"), float("-inf")], pd.NA).notna().all()]
    if bad:
        raise ValueError(f"Non-finite or non-numeric values in columns: {bad}")


def overlap_report(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, columns: Iterable[str]) -> dict:
    columns = list(columns)
    def keys(frame: pd.DataFrame) -> set[tuple]:
        return set(map(tuple, frame[columns].to_numpy().tolist()))
    train_keys, val_keys, test_keys = keys(train), keys(validation), keys(test)
    return {
        "train_validation_feature_overlap": len(train_keys & val_keys),
        "train_test_feature_overlap": len(train_keys & test_keys),
        "validation_test_feature_overlap": len(val_keys & test_keys),
    }
