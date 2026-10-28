"""Shared loading of inference artifacts for every serving entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from ids.features.preprocess import FeaturePreprocessor
from ids.pipeline.fuse import FusionPolicy
from ids.pipeline.predict import InferenceEngine
from ids.utils.io import read_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_fusion_policy(project_root: str | Path = PROJECT_ROOT, artifacts_dir: str | Path | None = None) -> FusionPolicy:
    project_root = Path(project_root)
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else project_root / "artifacts"
    candidates = [artifacts_dir / "thresholds" / "fusion.yaml", project_root / "configs" / "fusion.yaml"]
    config: dict[str, Any] = {}
    for path in candidates:
        if path.exists():
            config = read_yaml(path)
            break
    anomaly_threshold = config.get("anomaly_threshold")
    return FusionPolicy(
        known_confidence_threshold=float(config.get("known_confidence_threshold", 0.60)),
        anomaly_threshold=None if anomaly_threshold is None else float(anomaly_threshold),
        version=str(config.get("version", "1.0")),
    )


def load_model_version(artifacts_dir: str | Path, *, ready: bool) -> str:
    manifest_path = Path(artifacts_dir) / "metadata" / "inference_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return str(manifest.get("model_version", "local-artifacts"))
    return "local-artifacts" if ready else "untrained-development"


def load_inference_engine(
    artifacts_dir: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> InferenceEngine:
    project_root = Path(project_root)
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else project_root / "artifacts"
    preprocessor_path = artifacts_dir / "preprocess" / "preprocessor.joblib"
    svm_path = artifacts_dir / "models" / "svm_filter.joblib"
    policy = load_fusion_policy(project_root, artifacts_dir)
    if not preprocessor_path.exists() or not svm_path.exists():
        return InferenceEngine(policy=policy, model_version="untrained-development")

    preprocessor = FeaturePreprocessor.load(preprocessor_path)
    svm_filter = joblib.load(svm_path)
    stack_path = artifacts_dir / "models" / "stack_classifier.joblib"
    autoencoder_path = artifacts_dir / "models" / "autoencoder.pt.joblib"
    hmm_path = artifacts_dir / "models" / "temporal_hmm.joblib"
    stack = joblib.load(stack_path) if stack_path.exists() else None
    autoencoder = joblib.load(autoencoder_path) if autoencoder_path.exists() else None
    temporal_hmm = joblib.load(hmm_path) if hmm_path.exists() else None
    _validate_inference_manifest(
        artifacts_dir / "metadata" / "inference_manifest.json",
        preprocessor=preprocessor,
        svm_filter=svm_filter,
        stack=stack,
        temporal_hmm=temporal_hmm,
    )
    return InferenceEngine(
        preprocessor=preprocessor,
        svm_filter=svm_filter,
        stack_classifier=stack,
        autoencoder=autoencoder,
        temporal_hmm=temporal_hmm,
        policy=policy,
        model_version=load_model_version(artifacts_dir, ready=True),
    )


def _validate_inference_manifest(path: Path, *, preprocessor: Any, svm_filter: Any, stack: Any, temporal_hmm: Any) -> None:
    if not path.exists():
        raise RuntimeError(f"Inference manifest is missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Inference manifest cannot be read: {path}") from exc
    if not manifest.get("model_version"):
        raise RuntimeError("Inference manifest has no model_version")
    feature_columns = manifest.get("feature_columns")
    if feature_columns != list(preprocessor.columns):
        raise RuntimeError("Inference manifest feature schema does not match the preprocessor")
    labels = manifest.get("labels", [])
    if stack is not None and labels != [str(label) for label in stack.classes_]:
        raise RuntimeError("Inference manifest labels do not match the stack classifier")
    thresholds = manifest.get("thresholds", {})
    if "svm" in thresholds and abs(float(thresholds["svm"]) - float(svm_filter.threshold)) > 1e-9:
        raise RuntimeError("Inference manifest SVM threshold does not match the saved model")
    hmm_trained = thresholds.get("hmm_trained")
    if hmm_trained is True and temporal_hmm is None:
        raise RuntimeError("Manifest declares an HMM but no HMM artifact is available")
    if hmm_trained is False and temporal_hmm is not None:
        raise RuntimeError("An HMM artifact exists although the manifest marks HMM as unavailable")
