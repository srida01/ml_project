"""Leakage-resistant train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    manifest: dict


def _stratified_indices(labels: pd.Series, train_fraction: float, validation_fraction: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for _, group in labels.groupby(labels, dropna=False):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        n_train = max(1, int(len(indices) * train_fraction)) if len(indices) >= 3 else int(len(indices) * train_fraction)
        n_validation = int(len(indices) * validation_fraction)
        train.extend(indices[:n_train].tolist())
        validation.extend(indices[n_train:n_train + n_validation].tolist())
        test.extend(indices[n_train + n_validation:].tolist())
    return train, validation, test


def _group_counts(group_count: int, train_fraction: float, validation_fraction: float) -> tuple[int, int, int]:
    """Allocate at least one group to validation and test when possible."""
    if group_count < 3:
        return 0, 0, 0
    test_fraction = 1.0 - train_fraction - validation_fraction
    validation_count = max(1, int(round(group_count * validation_fraction)))
    test_count = max(1, int(round(group_count * test_fraction)))
    train_count = group_count - validation_count - test_count
    if train_count < 1:
        train_count = 1
        largest = "validation" if validation_count >= test_count else "test"
        if largest == "validation":
            validation_count = max(1, group_count - train_count - test_count)
        else:
            test_count = max(1, group_count - train_count - validation_count)
    return train_count, validation_count, test_count


def split_frame(
    frame: pd.DataFrame,
    *,
    label_column: str = "binary_label",
    group_columns: Iterable[str] = (),
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
    withheld_label_values: Iterable[str] = (),
    max_group_shuffles: int = 100,
) -> SplitResult:
    if abs(train_fraction + validation_fraction - 0.85) > 1e-9:
        raise ValueError("This splitter expects train + validation = 0.85, leaving 0.15 for test")
    if label_column not in frame:
        raise ValueError(f"Missing split label {label_column!r}")
    group_columns = [column for column in group_columns if column in frame.columns]
    withheld_label_values = {str(value) for value in withheld_label_values}
    if withheld_label_values and "multiclass_label" not in frame.columns:
        raise ValueError("Open-set withheld families require a multiclass_label column")
    withheld_mask = (
        frame["multiclass_label"].astype(str).isin(withheld_label_values)
        if withheld_label_values
        else pd.Series(False, index=frame.index)
    )
    groups = pd.Series("<row>", index=frame.index, dtype="string")
    train_mask = pd.Series(False, index=frame.index)
    validation_mask = pd.Series(False, index=frame.index)
    test_mask = pd.Series(False, index=frame.index)
    used_group_split = bool(group_columns and frame[group_columns].notna().any(axis=None))
    group_shuffle_attempts = 0
    if used_group_split:
        groups = frame[group_columns].astype("string").fillna("<missing>").agg("|".join, axis=1)
        unique_groups = groups.drop_duplicates().tolist()
        withheld_groups = set(groups[withheld_mask].drop_duplicates().tolist())
        eligible_groups = [group for group in unique_groups if group not in withheld_groups]
        import numpy as np

        rng = np.random.default_rng(seed)
        n_train, n_validation, _ = _group_counts(len(eligible_groups), train_fraction, validation_fraction)
        if n_train == 0:
            used_group_split = False
        else:
            # Whole capture files are often single-class (e.g. a benign-only
            # Monday). A partition without every target class cannot calibrate
            # thresholds or measure recall, so reshuffle deterministically
            # until each partition covers every class.
            labels = frame[label_column]
            required_classes = set(labels[~withheld_mask].dropna().unique().tolist())
            used_group_split = False
            for group_shuffle_attempts in range(1, max_group_shuffles + 1):
                order = list(eligible_groups)
                rng.shuffle(order)
                train_groups = set(order[:n_train])
                validation_groups = set(order[n_train:n_train + n_validation])
                test_groups = set(order[n_train + n_validation:]) | withheld_groups
                train_mask = groups.isin(train_groups)
                validation_mask = groups.isin(validation_groups)
                test_mask = groups.isin(test_groups)
                if all(
                    required_classes <= set(labels[mask & ~withheld_mask].dropna().unique().tolist())
                    for mask in (train_mask, validation_mask, test_mask)
                ):
                    used_group_split = True
                    break
    if not used_group_split:
        eligible = frame.loc[~withheld_mask]
        train_indices, validation_indices, test_indices = _stratified_indices(eligible[label_column], train_fraction, validation_fraction, seed)
        test_indices.extend(frame.index[withheld_mask].tolist())
        train_mask = frame.index.isin(train_indices)
        validation_mask = frame.index.isin(validation_indices)
        test_mask = frame.index.isin(test_indices)

    result = SplitResult(
        train=frame.loc[train_mask].reset_index(drop=True),
        validation=frame.loc[validation_mask].reset_index(drop=True),
        test=frame.loc[test_mask].reset_index(drop=True),
        manifest={
            "strategy": "group" if used_group_split else "row_stratified",
            "group_columns": group_columns,
            "seed": seed,
            "withheld_families": sorted(withheld_label_values),
            "group_shuffle_attempts": group_shuffle_attempts,
            "rows": {
                "train": int(train_mask.sum()),
                "validation": int(validation_mask.sum()),
                "test": int(test_mask.sum()),
            },
            "groups": {
                "train": sorted(set(groups[train_mask])) if used_group_split else [],
                "validation": sorted(set(groups[validation_mask])) if used_group_split else [],
                "test": sorted(set(groups[test_mask])) if used_group_split else [],
            },
        },
    )
    return result
