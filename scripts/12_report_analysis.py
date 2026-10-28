#!/usr/bin/env python3
"""Dataset statistics, model comparisons and figures for one finished experiment.

Run after ``run_pipeline.py``. Reads the frozen splits and saved artifacts, never
refits anything that the deployed engine uses, and writes:

- ``reports/tables/<tag>/analysis.json``  every number used in the report
- ``reports/figures/<tag>/*.pdf``         vector figures for LaTeX

Binary baselines (logistic regression, random forest, XGBoost) are trained here
on the same bounded, family-stratified training sample size the SVM gate uses,
so the comparison has an equal data budget.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from ids.data.labels import mapping_from_config
from ids.evaluation.metrics import classification_metrics
from ids.features.preprocess import FeaturePreprocessor
from ids.models.stack_classifier import _aligned_probabilities, _stratified_sample_indices
from ids.pipeline.load import load_inference_engine
from ids.utils.io import read_yaml, write_json

# Reference categorical palette (fixed order), sequential blue ramp and chart ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASE_MODEL_NAMES = {"random_forest": "Random forest", "xgboost": "XGBoost", "logistic_regression": "Logistic regression"}
ALERT_ORDER = ["benign", "anomaly_review", "suspicious_unclassified", "known_attack_suspected"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Experiment name used for output folders")
    parser.add_argument("--baseline-rows", type=int, default=100000)
    parser.add_argument("--per-family-rows", type=int, default=20000, help="Rows per family for the fused-engine breakdown")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    figures = Path("reports/figures") / args.tag
    tables = Path("reports/tables") / args.tag
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    result: dict = {"tag": args.tag}

    preprocessor = FeaturePreprocessor.load("artifacts/preprocess/preprocessor.joblib")
    columns = list(preprocessor.columns)
    result["feature_count"] = len(columns)
    keep = columns + ["binary_label", "multiclass_label", "source_file", "flow_id"]

    # ---- Dataset statistics -------------------------------------------------
    quality = json.loads(Path("data/manifests/data_quality_report.json").read_text())
    mapping = mapping_from_config(read_yaml("configs/label_map.yaml"))
    per_file = {}
    for item in quality["files"]:
        families: dict[str, int] = {}
        for raw, count in item["label_counts"].items():
            family = mapping.map_one(raw)
            families[family] = families.get(family, 0) + int(count)
        per_file[item["source_file"]] = {
            "rows": item["input_rows"],
            "families": families,
            "infinite_values": int(sum(item["infinite_values_replaced"].values())),
            "missing_after_cleaning": int(sum(v for k, v in item["missing_values"].items() if k in columns)),
            "exact_duplicate_rows": item["exact_duplicate_rows"],
            "constant_numeric_columns": item["constant_numeric_columns"],
        }
    result["per_file"] = per_file
    result["raw_label_counts"] = _sum_dicts(item["label_counts"] for item in quality["files"])
    result["total_rows"] = int(quality["total_output_rows"])
    result["split_manifest"] = json.loads(Path("data/processed/split_manifest.json").read_text())

    split_counts = {}
    for split in ("train", "validation", "test"):
        labels = pd.read_parquet(f"data/processed/{split}.parquet", columns=["multiclass_label"])["multiclass_label"]
        split_counts[split] = labels.value_counts().to_dict()
    result["split_family_counts"] = split_counts
    family_totals = _sum_dicts(split_counts.values())
    _family_distribution_figure(plt, family_totals, figures / "family_distribution.pdf")

    # ---- Load test set and transform once ----------------------------------
    test = pd.read_parquet("data/processed/test.parquet", columns=keep)
    X_test = preprocessor.transform(test)
    y_test = test["binary_label"].to_numpy()
    families_test = test["multiclass_label"].astype(str).to_numpy()

    # ---- Binary detection: baselines vs SVM gate ---------------------------
    train = pd.read_parquet("data/processed/train.parquet", columns=columns + ["binary_label", "multiclass_label"])
    sample_idx = _stratified_sample_indices(train["multiclass_label"].astype(str).to_numpy(), args.baseline_rows, args.seed)
    sample = train.iloc[np.sort(sample_idx)]
    del train
    X_sample, y_sample = preprocessor.transform(sample), sample["binary_label"].to_numpy()
    result["baseline_training_rows"] = int(len(sample))

    binary_rows, curves = [], {}
    for name, model in _binary_baselines(args.seed).items():
        start = time.perf_counter()
        model.fit(X_sample, y_sample)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        scores = model.predict_proba(X_test)[:, 1]
        predict_seconds = time.perf_counter() - start
        predictions = (scores >= 0.5).astype(int)
        binary_rows.append(_binary_row(name, y_test, predictions, scores, fit_seconds, predict_seconds))
        curves[name] = scores

    svm = joblib.load("artifacts/models/svm_filter.joblib")
    start = time.perf_counter()
    svm_scores = svm.score(X_test)
    predict_seconds = time.perf_counter() - start
    svm_predictions = (svm_scores >= svm.threshold).astype(int)
    binary_rows.append(_binary_row("SVM gate (validation-tuned)", y_test, svm_predictions, svm_scores, None, predict_seconds))
    curves["SVM gate (validation-tuned)"] = svm_scores
    system = json.loads(Path("reports/tables/test_system_metrics.json").read_text())
    fusion = system["fusion"]
    binary_rows.append({
        "model": "Full fused system", "precision": fusion["precision"], "recall": fusion["recall"], "f1": fusion["f1"],
        "fpr": fusion["fpr"], "roc_auc": None, "pr_auc": None, "fit_seconds": None,
        "ms_per_flow": system["inference_ms_per_flow"], "alerts_per_1000_benign": fusion["alert_volume_per_1000_benign"],
    })
    result["binary_comparison"] = binary_rows
    result["svm"] = {"model_name": svm.model_name, "threshold": float(svm.threshold)}
    result["system_metrics"] = system
    _roc_pr_figure(plt, y_test, curves, svm, figures / "roc_pr_binary.pdf")

    # Per-family detection rates for the gate (which attacks slip through).
    gate_by_family = pd.Series(svm_predictions).groupby(families_test).mean().to_dict()

    # ---- Autoencoder -------------------------------------------------------
    autoencoder_path = Path("artifacts/models/autoencoder.pt.joblib")
    ae_by_family = {}
    if autoencoder_path.exists():
        autoencoder = joblib.load(autoencoder_path)
        ae_scores = autoencoder.score(X_test)
        ae_flags = ae_scores >= autoencoder.threshold
        ae_by_family = pd.Series(ae_flags).groupby(families_test).mean().to_dict()
        result["autoencoder"] = {
            "threshold": float(autoencoder.threshold),
            "roc_auc": float(_safe_auc(y_test, ae_scores)),
            "benign_fpr": float(ae_flags[y_test == 0].mean()),
            "attack_tpr": float(ae_flags[y_test == 1].mean()),
        }
        _autoencoder_figure(plt, ae_scores, families_test, autoencoder.threshold, figures / "autoencoder_scores.pdf")
    result["detection_rate_by_family"] = {
        family: {"svm_gate": float(gate_by_family.get(family, np.nan)), "autoencoder": float(ae_by_family.get(family, np.nan)) if ae_by_family else None,
                 "test_rows": int((families_test == family).sum())}
        for family in sorted(set(families_test))
    }

    # ---- Known-attack family classification (true attack rows) -------------
    stack = joblib.load("artifacts/models/stack_classifier.joblib")
    attack_mask = y_test == 1
    X_attack, y_attack = X_test[attack_mask], families_test[attack_mask]
    known = np.isin(y_attack, stack.classes_)
    multiclass_rows, encoded = [], np.arange(len(stack.classes_))
    for name in stack.model_names:
        probabilities = _aligned_probabilities(stack.base_models[name], X_attack, encoded)
        predicted = stack.classes_[np.argmax(probabilities, axis=1)]
        multiclass_rows.append(_multiclass_row(BASE_MODEL_NAMES.get(name, name), y_attack, predicted, known))
    stack_predicted = stack.predict(X_attack)
    multiclass_rows.append(_multiclass_row("Stacked ensemble (OOF meta)", y_attack, stack_predicted, known))
    result["multiclass_comparison"] = multiclass_rows
    result["stack_per_class"] = classification_metrics(y_attack, stack_predicted)["per_class"]
    result["stack_classes"] = [str(c) for c in stack.classes_]
    result["unseen_test_families"] = sorted(set(y_attack) - set(stack.classes_))
    _confusion_figure(plt, y_attack, stack_predicted, figures / "stack_confusion.pdf")
    _importance_figure(plt, stack.base_models["random_forest"].feature_importances_, columns, figures / "rf_feature_importance.pdf")
    result["rf_top_features"] = [
        {"feature": columns[i], "importance": float(stack.base_models["random_forest"].feature_importances_[i])}
        for i in np.argsort(stack.base_models["random_forest"].feature_importances_)[::-1][:15]
    ]

    # ---- Fused engine: final alert per true family -------------------------
    engine = load_inference_engine(artifacts_dir="artifacts")
    rng = np.random.default_rng(args.seed)
    picks = np.concatenate([
        rng.choice(np.flatnonzero(families_test == family), size=min(args.per_family_rows, int((families_test == family).sum())), replace=False)
        for family in sorted(set(families_test))
    ])
    subset = test.iloc[np.sort(picks)]
    start = time.perf_counter()
    predicted = engine.predict_frame(subset[columns + ["flow_id"]])
    engine_seconds = time.perf_counter() - start
    crosstab = pd.crosstab(subset["multiclass_label"].astype(str).to_numpy(), predicted["final_alert_label"].to_numpy(), normalize="index")
    crosstab = crosstab.reindex(columns=[c for c in ALERT_ORDER if c in crosstab.columns] + [c for c in crosstab.columns if c not in ALERT_ORDER], fill_value=0.0)
    result["alert_breakdown_by_family"] = {family: row.to_dict() for family, row in crosstab.iterrows()}
    result["alert_breakdown_rows"] = {family: int((subset["multiclass_label"] == family).sum()) for family in crosstab.index}
    result["alert_level_counts"] = predicted["final_alert_level"].value_counts().to_dict()
    result["engine_ms_per_flow_subset"] = 1000.0 * engine_seconds / max(len(subset), 1)
    _alert_heatmap(plt, crosstab, figures / "alert_breakdown.pdf")
    _detection_figure(plt, result["detection_rate_by_family"], figures / "detection_by_family.pdf")

    write_json(result, tables / "analysis.json")
    print(f"Wrote {tables / 'analysis.json'} and figures in {figures}")


def _binary_baselines(seed: int) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    return {
        "Logistic regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist", random_state=seed, n_jobs=-1),
    }


def _binary_row(name, y_true, predictions, scores, fit_seconds, predict_seconds) -> dict:
    metrics = classification_metrics(y_true, predictions, scores)
    return {
        "model": name, "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"],
        "fpr": metrics["fpr"], "roc_auc": metrics.get("roc_auc"), "pr_auc": metrics.get("pr_auc"),
        "fit_seconds": fit_seconds, "ms_per_flow": 1000.0 * predict_seconds / len(y_true),
        "alerts_per_1000_benign": metrics["alert_volume_per_1000_benign"],
    }


def _multiclass_row(name, y_true, y_pred, known) -> dict:
    metrics = classification_metrics(y_true, y_pred)
    row = {"model": name, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"], "balanced_accuracy": metrics["balanced_accuracy"]}
    if known.any() and not known.all():
        row["accuracy_on_known_families"] = float((y_true[known] == y_pred[known]).mean())
    return row


def _safe_auc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return roc_auc_score(y_true, scores)
    except ValueError:
        return float("nan")


def _sum_dicts(dicts) -> dict:
    total: dict = {}
    for item in dicts:
        for key, value in item.items():
            total[key] = total.get(key, 0) + int(value)
    return total


# ---- Figures ------------------------------------------------------------------

def _style(plt) -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "savefig.bbox": "tight", "figure.dpi": 150,
    })


def _family_distribution_figure(plt, counts: dict, path: Path) -> None:
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    bars = ax.bar([k for k, _ in items], [v for _, v in items], color=SERIES[0], width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Flows (log scale)")
    ax.grid(axis="x", visible=False)
    for bar, (_, value) in zip(bars, items):
        ax.annotate(f"{value:,}", (bar.get_x() + bar.get_width() / 2, value), ha="center", va="bottom", fontsize=7, color=INK_MUTED, xytext=(0, 2), textcoords="offset points")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.savefig(path)
    plt.close(fig)


def _roc_pr_figure(plt, y_true, curves: dict, svm, path: Path) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for color, (name, scores) in zip(SERIES, curves.items()):
        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        roc_ax.plot(fpr, tpr, color=color, linewidth=1.6, label=name)
        pr_ax.plot(recall, precision, color=color, linewidth=1.6, label=name)
    roc_ax.set_xscale("log")
    roc_ax.set_xlim(1e-5, 1)
    roc_ax.set_xlabel("False positive rate (log)")
    roc_ax.set_ylabel("True positive rate")
    roc_ax.set_title("ROC", fontsize=9, color=INK)
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Precision-recall", fontsize=9, color=INK)
    handles, labels = roc_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(path)
    plt.close(fig)


def _autoencoder_figure(plt, scores, families, threshold, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    log_scores = np.log10(np.clip(scores, 1e-6, None))
    bins = np.linspace(np.percentile(log_scores, 0.1), np.percentile(log_scores, 99.9), 80)
    ax.hist(log_scores[families == "BENIGN"], bins=bins, color=SERIES[0], alpha=0.75, density=True, label="Benign")
    ax.hist(log_scores[families != "BENIGN"], bins=bins, color=SERIES[1], alpha=0.65, density=True, label="Attack")
    ax.axvline(np.log10(threshold), color=INK, linewidth=1.2, linestyle="--")
    ax.annotate("threshold (99th pct. benign val.)", (np.log10(threshold), ax.get_ylim()[1] * 0.9), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xlabel("log10 reconstruction error")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7.5)
    fig.savefig(path)
    plt.close(fig)


def _confusion_figure(plt, y_true, y_pred, path: Path) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    labels = sorted(set(y_true) | set(y_pred))
    matrix = pd.crosstab(pd.Categorical(y_true, labels), pd.Categorical(y_pred, labels), dropna=False, normalize="index").to_numpy()
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.imshow(matrix, cmap=LinearSegmentedColormap.from_list("seq", SEQUENTIAL), vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("True family")
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j] >= 0.005:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if matrix[i, j] > 0.55 else INK)
    fig.savefig(path)
    plt.close(fig)


def _importance_figure(plt, importances, columns, path: Path) -> None:
    order = np.argsort(importances)[::-1][:15][::-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.barh([columns[i] for i in order], importances[order], color=SERIES[0], height=0.6)
    ax.set_xlabel("Mean decrease in impurity")
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def _alert_heatmap(plt, crosstab: pd.DataFrame, path: Path) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    matrix = crosstab.to_numpy()
    ax.imshow(matrix, cmap=LinearSegmentedColormap.from_list("seq", SEQUENTIAL), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(crosstab.columns)), [c.replace("_", " ") for c in crosstab.columns], rotation=25, ha="right")
    ax.set_yticks(range(len(crosstab.index)), crosstab.index)
    ax.set_xlabel("Final alert label")
    ax.set_ylabel("True family")
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if matrix[i, j] > 0.55 else INK)
    fig.savefig(path)
    plt.close(fig)


def _detection_figure(plt, rates: dict, path: Path) -> None:
    families = [f for f in rates if f != "BENIGN"]
    x = np.arange(len(families))
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    ax.bar(x - 0.18, [rates[f]["svm_gate"] for f in families], width=0.34, color=SERIES[0], label="SVM gate")
    if rates[families[0]]["autoencoder"] is not None:
        ax.bar(x + 0.18, [rates[f]["autoencoder"] for f in families], width=0.34, color=SERIES[1], label="Autoencoder")
    ax.set_xticks(x, families, rotation=30, ha="right")
    ax.set_ylabel("Flows flagged")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
