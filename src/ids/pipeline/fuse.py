"""Transparent supervised/anomaly decision fusion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusionPolicy:
    known_confidence_threshold: float = 0.60
    anomaly_threshold: float | None = None
    version: str = "1.0"


def fuse_decision(*, svm_suspicious: bool, known_attack_label: str | None, known_attack_confidence: float | None, anomaly_flag: bool, policy: FusionPolicy) -> dict[str, str]:
    known_confident = bool(known_attack_label and (known_attack_confidence or 0.0) >= policy.known_confidence_threshold)
    if svm_suspicious and anomaly_flag and known_confident:
        return {"final_alert_label": "known_attack_suspected", "final_alert_level": "high"}
    if svm_suspicious and known_confident:
        return {"final_alert_label": "known_attack_suspected", "final_alert_level": "medium"}
    if anomaly_flag and not svm_suspicious:
        return {"final_alert_label": "anomaly_review", "final_alert_level": "medium"}
    if svm_suspicious:
        return {"final_alert_label": "suspicious_unclassified", "final_alert_level": "low"}
    return {"final_alert_label": "benign", "final_alert_level": "none"}
