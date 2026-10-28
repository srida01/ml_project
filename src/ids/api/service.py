"""Inference service used by all HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ids.pipeline.predict import InferenceEngine


@dataclass
class InferenceService:
    engine: InferenceEngine
    max_batch_rows: int = 10000

    @property
    def model_info(self) -> dict[str, Any]:
        labels = []
        if self.engine.stack_classifier is not None:
            labels = [str(label) for label in self.engine.stack_classifier.classes_]
        return {
            "model_version": self.engine.model_version,
            "required_features": self.engine.required_features,
            "supported_attack_labels": labels,
            "prediction_fields": list(self.engine.predict_frame(pd.DataFrame()).columns),
        }

    def predict_flow(self, flow: dict[str, Any]) -> dict[str, Any]:
        return self.predict_flows([flow])[0]

    def predict_flows(self, flows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not flows:
            raise ValueError("At least one flow is required")
        if len(flows) > self.max_batch_rows:
            raise ValueError(f"Batch exceeds maximum of {self.max_batch_rows} rows")
        frame = pd.DataFrame(flows)
        predictions = self.engine.predict_frame(frame)
        return to_json_records(predictions)


def to_json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert predictions to JSON-safe records with missing values as ``None``.

    ``where(..., None)`` alone keeps NaN in float columns, which JSON cannot
    encode, so cast to object first.
    """
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
