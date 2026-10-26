"""Inventory raw CSV files without modifying them."""

from __future__ import annotations

from pathlib import Path

from ids.utils.io import read_table, sha256_file, write_json


def inspect_path(path: str | Path) -> dict:
    path = Path(path)
    frame = read_table(path, nrows=1000)
    full_rows = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")) - 1 if path.suffix.lower() == ".csv" else None
    label_column = next((c for c in frame.columns if str(c).strip().lower() == "label"), None)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "columns": [str(c) for c in frame.columns],
        "column_count": len(frame.columns),
        "sample_rows": len(frame),
        "estimated_or_exact_rows": full_rows,
        "label_column": label_column,
        "label_counts_sample": frame[label_column].value_counts(dropna=False).to_dict() if label_column else {},
        "dtypes_sample": {str(k): str(v) for k, v in frame.dtypes.items()},
    }


def inspect_directory(input_dir: str | Path) -> dict:
    files = sorted(Path(input_dir).glob("*.csv"))
    return {"files": [inspect_path(path) for path in files]}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output")
    args = parser.parse_args()
    write_json(inspect_directory(args.input_dir), args.output)
