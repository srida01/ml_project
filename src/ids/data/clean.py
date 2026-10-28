"""Auditable deterministic cleaning for flow CSV files."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from ids.data.labels import LabelMapping, attach_targets
from ids.utils.io import read_table, sha256_file, write_json, write_table


NON_PREDICTIVE_COLUMNS = {
    "flow id",
    "src ip",
    "dst ip",
    "src port",
    "dst port",
    "protocol",
    "timestamp",
    "source file",
    "source row",
    "flow_id",
    "source_file",
    "source_row",
    "capture_day",
    "sequence_id",
    "duplicate_row",
}


def _is_non_predictive_column(column: str) -> bool:
    return str(column).strip().lower() in NON_PREDICTIVE_COLUMNS


def normalize_columns(columns: Iterable[object]) -> tuple[list[str], list[str]]:
    normalized: list[str] = []
    duplicates: list[str] = []
    seen: Counter[str] = Counter()
    for raw in columns:
        name = str(raw).replace("\ufeff", "").strip()
        seen[name] += 1
        if seen[name] > 1:
            duplicates.append(name)
            name = f"{name}__duplicate_{seen[name]}"
        normalized.append(name)
    return normalized, duplicates


def clean_frame(
    frame: pd.DataFrame,
    *,
    source_file: str = "unknown",
    mapping: LabelMapping | None = None,
    label_column: str = "Label",
    duplicate_policy: str = "preserve_and_flag",
    numeric_dtype: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    original_rows = len(frame)
    result = frame.copy()
    result.columns, duplicate_columns = normalize_columns(result.columns)
    if label_column not in result.columns and " Label" in result.columns:
        label_column = " Label"

    # Preserve identifiers and chronology as metadata. Converting these to
    # numbers would turn all-invalid columns into misleading all-NaN features.
    numeric_candidates = [c for c in result.columns if c != label_column and not _is_non_predictive_column(c)]
    coerced = {}
    inf_counts = {}
    for column in numeric_candidates:
        before_non_numeric = int(result[column].map(lambda value: isinstance(value, str)).sum())
        converted = pd.to_numeric(result[column], errors="coerce")
        inf_mask = converted.isin([float("inf"), float("-inf")])
        inf_counts[column] = int(inf_mask.sum())
        result[column] = converted.mask(inf_mask)
        if numeric_dtype is not None:
            # float32 halves memory for the ~2.8M-row CICIDS2017 table.
            result[column] = result[column].astype(numeric_dtype)
        coerced[column] = before_non_numeric

    duplicate_mask = result.duplicated(keep=False)
    duplicate_rows = int(result.duplicated().sum())
    if duplicate_policy == "error" and duplicate_rows:
        raise ValueError(f"Found {duplicate_rows} duplicate row(s) in {source_file}")
    if duplicate_policy == "preserve_and_flag":
        result["duplicate_row"] = duplicate_mask.astype("int8")
    elif duplicate_policy not in {"preserve", "error"}:
        raise ValueError(f"Unsupported duplicate policy: {duplicate_policy!r}")

    if "source_file" in result.columns:
        result["source_file"] = source_file
    else:
        result.insert(0, "source_file", source_file)
    if "source_row" in result.columns:
        result["source_row"] = range(original_rows)
    else:
        result.insert(1, "source_row", range(original_rows))
    result["flow_id"] = result["source_file"].astype(str) + ":" + result["source_row"].astype(str)
    if mapping is not None:
        result = attach_targets(result, mapping, label_column=label_column)

    numeric = result.select_dtypes(include="number")
    constant_columns = [column for column in numeric.columns if numeric[column].nunique(dropna=False) <= 1]
    report = {
        "source_file": source_file,
        "input_rows": original_rows,
        "output_rows": len(result),
        "dropped_rows": original_rows - len(result),
        "duplicate_columns": duplicate_columns,
        "invalid_numeric_values_coerced": coerced,
        "infinite_values_replaced": inf_counts,
        "missing_values": {str(k): int(v) for k, v in result.isna().sum().items()},
        "constant_numeric_columns": constant_columns,
        "exact_duplicate_rows": duplicate_rows,
        "duplicate_flagged_rows": int(duplicate_mask.sum()),
        "label_counts": result[label_column].value_counts(dropna=False).to_dict() if label_column in result else {},
        "sha256": None,
    }
    return result, report


def clean_directory(
    input_dir: str | Path,
    output: str | Path,
    *,
    mapping: LabelMapping | None = None,
    label_column: str = "Label",
    report_path: str | Path | None = None,
    duplicate_policy: str = "preserve_and_flag",
    encoding: str = "utf-8",
    numeric_dtype: str | None = None,
) -> dict:
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    frames = []
    reports = []
    for path in files:
        raw = read_table(path, encoding=encoding)
        cleaned, report = clean_frame(
            raw,
            source_file=path.name,
            mapping=mapping,
            label_column=label_column,
            duplicate_policy=duplicate_policy,
            numeric_dtype=numeric_dtype,
        )
        report["sha256"] = sha256_file(path)
        frames.append(cleaned)
        reports.append(report)
    combined = pd.concat(frames, ignore_index=True)
    write_table(combined, output)
    summary = {
        "files": reports,
        "total_input_rows": sum(item["input_rows"] for item in reports),
        "total_output_rows": len(combined),
        "output": str(output),
    }
    if report_path:
        write_json(summary, report_path)
    return summary
