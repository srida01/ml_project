"""Versioned artifact manifest helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ids.utils.io import write_json


def write_inference_manifest(path: str | Path, *, model_version: str, feature_columns: list[str], labels: list[str], thresholds: dict[str, Any], notes: str = "") -> dict[str, Any]:
    manifest = {
        "model_version": model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feature_columns,
        "labels": labels,
        "thresholds": thresholds,
        "notes": notes,
    }
    write_json(manifest, path)
    return manifest
