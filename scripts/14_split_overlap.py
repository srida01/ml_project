#!/usr/bin/env python3
"""Measure exact feature-vector overlap between split partitions.

A test flow whose complete feature vector also occurs in the training split can
be "recognised" rather than generalised to, so this overlap is a direct measure
of how optimistic a split is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from ids.features.preprocess import FeaturePreprocessor
from ids.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--preprocessor", default="artifacts/preprocess/preprocessor.joblib")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    columns = list(FeaturePreprocessor.load(args.preprocessor).columns)
    hashes, families = {}, {}
    for split in ("train", "validation", "test"):
        frame = pd.read_parquet(Path(args.processed_dir) / f"{split}.parquet", columns=columns + ["multiclass_label"])
        hashes[split] = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()
        families[split] = frame["multiclass_label"].astype(str).to_numpy()
        del frame
    train_hashes = np.unique(hashes["train"])
    in_train = np.isin(hashes["test"], train_hashes)
    by_family = pd.Series(in_train).groupby(families["test"]).mean().to_dict()
    result = {
        "test_rows": int(len(in_train)),
        "test_rows_with_exact_copy_in_train": int(in_train.sum()),
        "test_fraction_with_exact_copy_in_train": float(in_train.mean()),
        "validation_fraction_with_exact_copy_in_train": float(np.isin(hashes["validation"], train_hashes).mean()),
        "test_fraction_by_family": {family: float(value) for family, value in by_family.items()},
    }
    write_json(result, args.output)
    print(result)


if __name__ == "__main__":
    main()
