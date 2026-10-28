#!/usr/bin/env python3
"""Run the reproducible IDS workflow from raw CSVs to saved model artifacts.

This is an orchestration script, not a replacement for the individual stage
scripts. Each stage remains independently rerunnable and writes its own output.
The script stops on required-stage failures and reports optional HMM absence as
an explicit limitation because many CICIDS2017 releases have no sequence ID.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run_stage(name: str, *arguments: str, allow_failure: bool = False) -> bool:
    command = [sys.executable, str(SCRIPTS / name), *arguments]
    print(f"\n=== {name} ===")
    print("$ " + " ".join(command))
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(command, cwd=ROOT, env=environment)
    if completed.returncode != 0 and not allow_failure:
        raise SystemExit(f"Stage failed: {name} (exit code {completed.returncode})")
    return completed.returncode == 0


def evaluate_test_set() -> None:
    """Evaluate the frozen test set without changing any model parameters."""
    import joblib

    sys.path.insert(0, str(ROOT / "src"))
    from ids.evaluation.metrics import classification_metrics
    from ids.features.preprocess import FeaturePreprocessor
    from ids.pipeline.load import load_inference_engine
    from ids.utils.io import read_table, write_json

    test = read_table(ROOT / "data/processed/test.parquet")
    preprocessor = FeaturePreprocessor.load(ROOT / "artifacts/preprocess/preprocessor.joblib")
    svm_filter = joblib.load(ROOT / "artifacts/models/svm_filter.joblib")
    scores = svm_filter.score(preprocessor.transform(test))
    predictions = svm_filter.predict(preprocessor.transform(test))
    metrics = classification_metrics(test["binary_label"], predictions, scores)
    output = ROOT / "reports/tables/test_binary_metrics.json"
    write_json(metrics, output)
    print(f"Wrote frozen test metrics to {output}")

    # Evaluate the complete saved inference contract as well.  Targets are
    # removed before prediction so the production path cannot accidentally
    # consume labels or evaluation-only columns.
    engine = load_inference_engine(artifacts_dir=ROOT / "artifacts", project_root=ROOT)
    inference_frame = test.drop(
        columns=["Label", "raw_label", "binary_label", "multiclass_label", "predicted_binary_label"],
        errors="ignore",
    )
    inference_start = time.perf_counter()
    predicted = engine.predict_frame(inference_frame)
    inference_seconds = time.perf_counter() - inference_start
    system_metrics = {
        "svm_gate": metrics,
        "fusion": classification_metrics(
            test["binary_label"],
            (predicted["final_alert_label"] != "benign").astype("int8"),
            predicted["svm_score"],
        ),
        "rows": len(test),
        "temporal_rows": int(predicted["hmm_current_state"].notna().sum()),
        "inference_seconds": inference_seconds,
        "inference_ms_per_flow": 1000.0 * inference_seconds / max(len(test), 1),
        "inference_flows_per_second": len(test) / max(inference_seconds, 1e-12),
    }
    attack_mask = test["binary_label"].to_numpy() == 1
    if attack_mask.any() and predicted.loc[attack_mask, "known_attack_label"].notna().any():
        oracle_true = test.loc[attack_mask, "multiclass_label"]
        oracle_pred = predicted.loc[attack_mask, "known_attack_label"].fillna("SVM_GATE_MISS")
        system_metrics["known_attack_cascade"] = classification_metrics(oracle_true, oracle_pred)
    if predicted["anomaly_score"].notna().any():
        system_metrics["autoencoder"] = {
            "benign_fpr": float(predicted.loc[test["binary_label"] == 0, "anomaly_flag"].mean()),
            "attack_tpr": float(predicted.loc[test["binary_label"] == 1, "anomaly_flag"].mean()),
        }
    system_output = ROOT / "reports/tables/test_system_metrics.json"
    write_json(system_metrics, system_output)
    print(f"Wrote frozen system metrics to {system_output}")


def write_inference_manifest(hmm_trained: bool) -> str:
    import joblib

    sys.path.insert(0, str(ROOT / "src"))
    from ids.features.preprocess import FeaturePreprocessor
    from ids.pipeline.load import load_fusion_policy
    from ids.utils.artifacts import write_inference_manifest as write_manifest

    preprocessor = FeaturePreprocessor.load(ROOT / "artifacts/preprocess/preprocessor.joblib")
    svm_filter = joblib.load(ROOT / "artifacts/models/svm_filter.joblib")
    stack_path = ROOT / "artifacts/models/stack_classifier.joblib"
    stack = joblib.load(stack_path) if stack_path.exists() else None
    autoencoder_path = ROOT / "artifacts/models/autoencoder.pt.joblib"
    autoencoder = joblib.load(autoencoder_path) if autoencoder_path.exists() else None
    policy = load_fusion_policy(ROOT, ROOT / "artifacts")
    model_version = "pipeline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_manifest(
        ROOT / "artifacts/metadata/inference_manifest.json",
        model_version=model_version,
        feature_columns=preprocessor.columns,
        labels=[] if stack is None else [str(label) for label in stack.classes_],
        thresholds={
            "svm": float(svm_filter.threshold),
            "autoencoder": None if autoencoder is None else float(autoencoder.threshold),
            "fusion": {
                "known_confidence_threshold": policy.known_confidence_threshold,
                "anomaly_threshold": policy.anomaly_threshold,
                "version": policy.version,
            },
            "hmm_trained": hmm_trained,
        },
        notes="Generated by scripts/run_pipeline.py",
    )
    return model_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing unchanged CICIDS2017 CSV files")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip the simple logistic/random-forest baselines")
    parser.add_argument("--skip-autoencoder", action="store_true", help="Skip the PyTorch autoencoder")
    parser.add_argument("--skip-hmm", action="store_true", help="Skip exploratory HMM training")
    args = parser.parse_args()

    raw_dir = ROOT / args.raw_dir
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {raw_dir}. Put CICIDS2017 raw files there first.")

    run_stage("00_inspect.py", "--input-dir", str(args.raw_dir))
    run_stage("01_clean.py", "--input-dir", str(args.raw_dir))
    run_stage("02_split.py")
    run_stage("03_fit_preprocessor.py")

    if not args.skip_baselines:
        run_stage("04_train_baselines.py")
    run_stage("05_train_svm.py")
    run_stage("06_train_stack.py")

    if not args.skip_autoencoder:
        run_stage("07_train_autoencoder.py")
    else:
        (ROOT / "artifacts/models/autoencoder.pt").unlink(missing_ok=True)
        (ROOT / "artifacts/models/autoencoder.pt.joblib").unlink(missing_ok=True)

    hmm_trained = False
    if not args.skip_hmm:
        # HMM is optional because the downloaded CICIDS2017 CSVs may not have
        # valid sequence IDs/timestamps. Do not make the rest of the pipeline
        # look temporally valid when the source data cannot support it.
        run_stage(
            "08_train_hmm.py",
            "--preprocessor",
            "artifacts/preprocess/preprocessor.joblib",
            "--svm",
            "artifacts/models/svm_filter.joblib",
            "--stack",
            "artifacts/models/stack_classifier.joblib",
            "--autoencoder",
            "" if args.skip_autoencoder else "artifacts/models/autoencoder.pt.joblib",
        )
        hmm_trained = (ROOT / "artifacts/models/temporal_hmm.joblib").exists()
    else:
        (ROOT / "artifacts/models/temporal_hmm.joblib").unlink(missing_ok=True)

    run_stage("09_select_fusion.py")
    model_version = write_inference_manifest(hmm_trained)
    evaluate_test_set()

    summary = {
        "status": "completed",
        "raw_files": [path.name for path in csv_files],
        "cleaned_data": "data/interim/cicids2017_cleaned.parquet",
        "splits": "data/processed/{train,validation,test}.parquet",
        "artifacts": "artifacts/",
        "test_metrics": "reports/tables/test_binary_metrics.json",
        "model_version": model_version,
        "inference_manifest": "artifacts/metadata/inference_manifest.json",
        "hmm_trained": hmm_trained,
        "hmm": "optional; unavailable when no valid sequence metadata exists",
    }
    print("\nPipeline complete:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
