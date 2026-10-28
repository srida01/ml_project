"""Leakage-agnostic metrics with safe handling of degenerate labels."""

from __future__ import annotations

from typing import Iterable


def classification_metrics(y_true: Iterable, y_pred: Iterable, scores: Iterable | None = None, positive_label=1) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
    )

    y_true = list(y_true)
    y_pred = list(y_pred)
    labels = list(dict.fromkeys(y_true + y_pred))
    precision_by_class, recall_by_class, f1_by_class, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    if len(labels) == 2:
        selected_positive = positive_label if positive_label in labels else labels[-1]
        precision = precision_score(y_true, y_pred, pos_label=selected_positive, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=selected_positive, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=selected_positive, zero_division=0)
    else:
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": matrix,
        "per_class": {
            str(label): {
                "precision": float(precision_by_class[index]),
                "recall": float(recall_by_class[index]),
                "f1": float(f1_by_class[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }
    if len(labels) == 2:
        selected_positive = positive_label if positive_label in labels else labels[-1]
        negative_labels = [label for label in labels if label != selected_positive]
        negative = negative_labels[0] if negative_labels else None
        counts = binary_counts(y_true, y_pred, positive_label=selected_positive)
        result["specificity"] = float(counts["tn"] / (counts["tn"] + counts["fp"])) if negative is not None and counts["tn"] + counts["fp"] else 0.0
        result["fpr"] = float(counts["fp"] / (counts["tn"] + counts["fp"])) if negative is not None and counts["tn"] + counts["fp"] else 0.0
        result["alert_volume_per_1000_benign"] = float(1000.0 * counts["fp"] / (counts["tn"] + counts["fp"])) if negative is not None and counts["tn"] + counts["fp"] else 0.0
    if scores is not None and len(labels) == 2:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, list(scores)))
            result["pr_auc"] = float(average_precision_score(y_true, list(scores)))
        except ValueError:
            pass
    return result


def binary_counts(y_true: Iterable, y_pred: Iterable, positive_label=1) -> dict[str, int]:
    y_true, y_pred = list(y_true), list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return {
        "tp": sum(actual == positive_label and predicted == positive_label for actual, predicted in zip(y_true, y_pred)),
        "tn": sum(actual != positive_label and predicted != positive_label for actual, predicted in zip(y_true, y_pred)),
        "fp": sum(actual != positive_label and predicted == positive_label for actual, predicted in zip(y_true, y_pred)),
        "fn": sum(actual == positive_label and predicted != positive_label for actual, predicted in zip(y_true, y_pred)),
    }
