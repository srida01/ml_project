"""The single prediction implementation shared by CLI, API, and dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import numpy as np

from ids.pipeline.fuse import FusionPolicy, fuse_decision


@dataclass
class InferenceEngine:
    preprocessor: Any = None
    svm_filter: Any = None
    stack_classifier: Any = None
    autoencoder: Any = None
    temporal_hmm: Any = None
    policy: FusionPolicy = FusionPolicy()
    model_version: str = "untrained-development"

    @property
    def required_features(self) -> list[str]:
        return list(self.preprocessor.columns) if self.preprocessor is not None else []

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=prediction_columns())
        if self.preprocessor is None:
            raise RuntimeError("No fitted preprocessor is loaded")
        missing = [column for column in self.required_features if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        X = self.preprocessor.transform(frame)
        if self.svm_filter is None:
            raise RuntimeError("No binary SVM filter is loaded")
        svm_scores = self.svm_filter.score(X)
        svm_suspicious = self.svm_filter.predict(X).astype(bool)
        if self.stack_classifier is not None:
            labels, confidences = self.stack_classifier.predict_with_confidence(X)
        else:
            labels = [None] * len(frame)
            confidences = [None] * len(frame)
        anomaly_scores = self.autoencoder.score(X) if self.autoencoder is not None else [None] * len(frame)
        anomaly_threshold = self.policy.anomaly_threshold
        if anomaly_threshold is None:
            anomaly_threshold = getattr(self.autoencoder, "threshold", None)
        anomaly_flags = [bool(score >= anomaly_threshold) if score is not None and anomaly_threshold is not None else False for score in anomaly_scores]
        rows = []
        for index in range(len(frame)):
            known_label = None if labels[index] is None else str(labels[index])
            confidence_value = confidences[index]
            known_confidence = float(confidence_value) if confidence_value is not None else None
            score_value = anomaly_scores[index]
            score = float(score_value) if score_value is not None else None
            fused = fuse_decision(svm_suspicious=bool(svm_suspicious[index]), known_attack_label=known_label, known_attack_confidence=known_confidence, anomaly_flag=anomaly_flags[index], policy=self.policy)
            rows.append({
                "flow_id": str(frame.iloc[index].get("flow_id", index)),
                "svm_score": float(svm_scores[index]),
                "svm_suspicious": bool(svm_suspicious[index]),
                "known_attack_label": known_label if bool(svm_suspicious[index]) else None,
                "known_attack_confidence": known_confidence if bool(svm_suspicious[index]) else None,
                "anomaly_score": score,
                "anomaly_flag": anomaly_flags[index],
                **fused,
                "sequence_id": frame.iloc[index].get("sequence_id"),
                "hmm_current_state": None,
                "hmm_next_state": None,
                "hmm_next_state_confidence": None,
                "model_version": self.model_version,
                "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            })
        if self.temporal_hmm is not None and "sequence_id" in frame.columns:
            histories: dict[object, list[list[float]]] = {}
            configured_emissions = getattr(self.temporal_hmm, "emission_columns", None)
            emission_columns = list(configured_emissions) if configured_emissions else []
            for index, output in enumerate(rows):
                sequence_id = frame.iloc[index].get("sequence_id")
                if sequence_id is None or pd.isna(sequence_id) or not emission_columns:
                    continue
                observation = []
                for column in emission_columns:
                    value = output.get(column)
                    observation.append(0.0 if value is None or pd.isna(value) else float(value))
                history = histories.setdefault(sequence_id, [])
                history.append(observation)
                context = self.temporal_hmm.predict_context(np.asarray(history, dtype=float))
                output["hmm_current_state"] = context["current_state"]
                output["hmm_next_state"] = context["next_state"]
                output["hmm_next_state_confidence"] = context["next_state_confidence"]
        return pd.DataFrame(rows, columns=prediction_columns())


def prediction_columns() -> list[str]:
    return ["flow_id", "svm_score", "svm_suspicious", "known_attack_label", "known_attack_confidence", "anomaly_score", "anomaly_flag", "final_alert_label", "final_alert_level", "sequence_id", "hmm_current_state", "hmm_next_state", "hmm_next_state_confidence", "model_version", "inference_timestamp"]
