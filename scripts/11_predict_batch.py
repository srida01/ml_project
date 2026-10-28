#!/usr/bin/env python3
"""Run saved inference on CSV/Parquet input."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from ids.features.preprocess import FeaturePreprocessor
from ids.pipeline.load import _validate_inference_manifest, load_fusion_policy, load_model_version
from ids.pipeline.predict import InferenceEngine
from ids.utils.io import read_table, write_table


def load_engine(
    preprocessor_path: str,
    svm_path: str,
    stack_path: str | None = None,
    autoencoder_path: str | None = None,
    hmm_path: str | None = None,
    artifacts_dir: str = "artifacts",
) -> InferenceEngine:
    import joblib

    preprocessor = FeaturePreprocessor.load(preprocessor_path)
    svm_filter = joblib.load(svm_path)
    stack = joblib.load(stack_path) if stack_path and Path(stack_path).is_file() else None
    autoencoder = joblib.load(autoencoder_path) if autoencoder_path and Path(autoencoder_path).is_file() else None
    temporal_hmm = joblib.load(hmm_path) if hmm_path and Path(hmm_path).is_file() else None
    artifacts_path = Path(artifacts_dir)
    engine = InferenceEngine(
        preprocessor=preprocessor,
        svm_filter=svm_filter,
        stack_classifier=stack,
        autoencoder=autoencoder,
        temporal_hmm=temporal_hmm,
        policy=load_fusion_policy(Path(__file__).resolve().parents[1], artifacts_path),
        model_version=load_model_version(artifacts_path, ready=True),
    )
    _validate_inference_manifest(
        artifacts_path / "metadata" / "inference_manifest.json",
        preprocessor=preprocessor,
        svm_filter=svm_filter,
        stack=stack,
        temporal_hmm=temporal_hmm,
    )
    return engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preprocessor", default="artifacts/preprocess/preprocessor.joblib")
    parser.add_argument("--svm", default="artifacts/models/svm_filter.joblib")
    parser.add_argument("--stack", default="artifacts/models/stack_classifier.joblib")
    parser.add_argument("--autoencoder", default="artifacts/models/autoencoder.pt.joblib")
    parser.add_argument("--hmm", default="artifacts/models/temporal_hmm.joblib")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()
    engine = load_engine(args.preprocessor, args.svm, args.stack, args.autoencoder, args.hmm, args.artifacts_dir)
    write_table(engine.predict_frame(read_table(args.input)), args.output)
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
