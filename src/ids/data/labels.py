"""Explicit, versioned CICIDS label transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd


@dataclass(frozen=True)
class LabelMapping:
    benign_label: str
    family_map: Mapping[str, str]
    unknown_policy: str = "error"

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.family_map.values()))

    def map_one(self, raw_label: object) -> str:
        label = str(raw_label).strip()
        if label in self.family_map:
            return self.family_map[label]
        # Some CICIDS2017 distributions append " - Attempted" to labels.
        # Keep the original value in raw_label, but map the documented suffix
        # to the same attack family when the base label is explicitly known.
        attempted_base = label.removesuffix(" - Attempted")
        if attempted_base != label and attempted_base in self.family_map:
            return self.family_map[attempted_base]
        if self.unknown_policy == "other":
            return "OTHER"
        raise ValueError(f"Unmapped raw label: {label!r}")

    def transform(self, labels: Iterable[object]) -> pd.DataFrame:
        raw_values = [str(value).strip() for value in labels]
        raw = pd.Series(raw_values, name="raw_label", dtype="string")
        multiclass = raw.map(self.map_one)
        return pd.DataFrame(
            {
                "raw_label": raw,
                "binary_label": (raw != self.benign_label).astype("int8"),
                "multiclass_label": multiclass,
            }
        )


def mapping_from_config(config: Mapping[str, object]) -> LabelMapping:
    raw_family_map = config.get("family_map", {})
    family_map = raw_family_map if isinstance(raw_family_map, Mapping) else {}
    return LabelMapping(
        benign_label=str(config.get("benign_label", "BENIGN")),
        family_map={str(k): str(v) for k, v in family_map.items()},
        unknown_policy=str(config.get("unknown_policy", "error")),
    )


def attach_targets(frame: pd.DataFrame, mapping: LabelMapping, label_column: str = "Label") -> pd.DataFrame:
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column {label_column!r}")
    result = frame.copy()
    targets = mapping.transform(result[label_column])
    result["raw_label"] = targets["raw_label"].to_numpy()
    result["binary_label"] = targets["binary_label"].to_numpy()
    result["multiclass_label"] = targets["multiclass_label"].to_numpy()
    return result
