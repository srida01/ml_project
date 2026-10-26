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
    return InferenceEngine(
        preprocessor=preprocessor,
        svm_filter=svm_filter,
        stack_classifier=stack,
        autoencoder=autoencoder,
        temporal_hmm=temporal_hmm,
        policy=policy,
        model_version=load_model_version(artifacts_dir, ready=True),
    )
