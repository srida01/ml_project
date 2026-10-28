#!/usr/bin/env python3
"""Create frozen train/validation/test tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from ids.data.split import split_frame
from ids.utils.io import read_table, read_yaml, write_json, write_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/cicids2017_cleaned.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--config", default="configs/split.yaml")
    args = parser.parse_args()
    config = read_yaml(args.config)
    frame = read_table(args.input)
    group_columns = list(config.get("group_columns", []))
    # A sequence must never be divided across partitions.  Include the
    # sequence identity when it is available, while retaining capture/source
    # context so IDs reused on different days remain distinct.
    if "sequence_id" in frame.columns and "sequence_id" not in group_columns:
        group_columns.append("sequence_id")
    result = split_frame(frame, label_column="binary_label", group_columns=group_columns, train_fraction=float(config.get("train_fraction", 0.70)), validation_fraction=float(config.get("validation_fraction", 0.15)), seed=int(config.get("seed", 42)), withheld_label_values=config.get("open_set_withheld_families", []))
    output_dir = Path(args.output_dir)
    write_table(result.train, output_dir / "train.parquet")
    write_table(result.validation, output_dir / "validation.parquet")
    write_table(result.test, output_dir / "test.parquet")
    write_json(result.manifest, output_dir / "split_manifest.json")
    print(f"Wrote split rows: {result.manifest['rows']}")


if __name__ == "__main__":
    main()
