#!/usr/bin/env python3
"""Clean raw flow files without overwriting raw data."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from ids.data.clean import clean_directory
from ids.data.labels import mapping_from_config
from ids.utils.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output", default="data/interim/cicids2017_cleaned.parquet")
    parser.add_argument("--report", default="data/manifests/data_quality_report.json")
    parser.add_argument("--label-map", default="configs/label_map.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()
    data_config = read_yaml(args.data_config)
    summary = clean_directory(
        args.input_dir,
        args.output,
        mapping=mapping_from_config(read_yaml(args.label_map)),
        label_column=str(data_config.get("label_column", "Label")),
        report_path=args.report,
        duplicate_policy=str(data_config.get("duplicate_policy", "preserve_and_flag")),
        encoding=str(data_config.get("encoding", "utf-8")),
        numeric_dtype=data_config.get("numeric_dtype"),
    )
    print(f"Cleaned {summary['total_input_rows']} rows from {len(summary['files'])} file(s); wrote {args.output}")


if __name__ == "__main__":
    main()
