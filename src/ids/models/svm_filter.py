"""High-recall binary suspicious-flow filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SVMFilter:
    model: Any
    threshold: float = 0.0
    model_name: str = "rbf_svm"

    @classmethod
    def fit(
        cls,
        X,
        y,
        *,
        kernel: str = "rbf",
        C: float = 2.0,
        gamma: str | float = "scale",
        class_weight: str | dict | None = "balanced",
        max_exact_rows: int = 100000,
        approx_components: int = 512,
        transform_batch_size: int = 4096,
        seed: int = 42,
    ) -> "SVMFilter":
        from sklearn.svm import SVC

        X_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y)
        X_fit, y_fit = X_array, y_array
        model_name = f"{kernel}_svm"
        if kernel != "linear" and len(X_fit) > max_exact_rows:
            from sklearn.kernel_approximation import RBFSampler
            from sklearn.linear_model import SGDClassifier

            # Exact kernel SVMs and a full RBFSampler projection are both
            # memory-prohibitive at CICIDS2017 scale. Fit the approximation on
            # a bounded, stratified sample and transform inference in batches.
            sampled_indices = _stratified_sample_indices(y_array, max_exact_rows, seed)
            X_fit = X_array[sampled_indices]
            y_fit = y_array[sampled_indices]
            if gamma == "scale":
                variance = float(np.var(X_fit))
                gamma_value = 1.0 / max(X_fit.shape[1] * variance, np.finfo(np.float32).eps)
            else:
                gamma_value = gamma
            sampler = RBFSampler(
                gamma=gamma_value,
                n_components=min(approx_components, max(64, X_fit.shape[1] * 8)),
                random_state=seed,
            )
            # Do not call ``fit_transform`` here.  RBFSampler materializes the
            # complete projected matrix, which can be many gigabytes even for
            # the bounded sample.  Fit the random projection once, then feed
            # projected batches directly to the incremental linear classifier.
            sampler.fit(X_fit)
            # ``SGDClassifier.partial_fit`` does not accept
            # ``class_weight="balanced"`` directly, so convert the requested
            # class weighting into per-row weights once for the sampled data.
            if class_weight is None:
                sample_weights = None
            else:
                from sklearn.utils.class_weight import compute_sample_weight

                sample_weights = compute_sample_weight(class_weight, y_fit)
            model = SGDClassifier(loss="hinge", class_weight=None, random_state=seed)
            classes = np.unique(y_fit)
            for start in range(0, len(X_fit), transform_batch_size):
                stop = start + transform_batch_size
                transformed = np.asarray(sampler.transform(X_fit[start:stop]), dtype=np.float32)
                model.partial_fit(
                    transformed,
                    y_fit[start:stop],
                    classes=classes,
                    sample_weight=None if sample_weights is None else sample_weights[start:stop],
                )
            return cls(_ApproximateModel(sampler, model, transform_batch_size), threshold=0.0, model_name="rbf_sampler_linear_svm")
        model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight, probability=True, random_state=seed)
        model.fit(X_array, y_array)
        return cls(model, threshold=0.0, model_name=model_name)

    def score(self, X) -> np.ndarray:
        if hasattr(self.model, "decision_function"):
            return np.asarray(self.model.decision_function(X), dtype=float)
        probabilities = self.model.predict_proba(X)
        return np.asarray(probabilities[:, 1], dtype=float)

    def predict(self, X) -> np.ndarray:
        return (self.score(X) >= self.threshold).astype("int8")

    def select_high_recall_threshold(self, X_validation, y_validation, target_recall: float = 0.98) -> float:
        scores = self.score(X_validation)
        candidates = np.unique(scores)
        best = float(np.min(candidates)) if len(candidates) else 0.0
        if not len(candidates):
            self.threshold = best
            return best

        # Recall changes only when the threshold crosses a positive score.
        # Selecting from every candidate and recomputing recall scans the full
        # validation set once per distinct score (quadratic at CICIDS scale).
        # The highest threshold meeting the target is therefore the score of
        # the ceil(target_recall * positive_count)-th highest positive sample.
        positive_scores = np.sort(scores[np.asarray(y_validation) == 1])[::-1]
        if len(positive_scores) == 0:
            # Falling back to the minimum score would flag every flow.
            raise ValueError("Validation data contains no attack rows; the SVM recall threshold cannot be calibrated")
        required = int(np.ceil(target_recall * len(positive_scores)))
        if required <= 0:
            best = float(candidates[-1])
        elif required <= len(positive_scores):
            best = float(positive_scores[required - 1])
        self.threshold = best
        return best

    def predict_proba(self, X) -> np.ndarray:
        scores = self.score(X)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
        return np.column_stack([1.0 - probabilities, probabilities])


class _ApproximateModel:
    def __init__(self, sampler, classifier, transform_batch_size: int = 4096):
        self.sampler = sampler
        self.classifier = classifier
        self.transform_batch_size = transform_batch_size

    def decision_function(self, X):
        values = np.asarray(X, dtype=np.float32)
        scores = []
        for start in range(0, len(values), self.transform_batch_size):
            batch = values[start:start + self.transform_batch_size]
            transformed = np.asarray(self.sampler.transform(batch), dtype=np.float32)
            scores.append(np.asarray(self.classifier.decision_function(transformed), dtype=float))
        return np.concatenate(scores) if scores else np.empty(0, dtype=float)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
        return np.column_stack([1 - p, p])


def _stratified_sample_indices(labels: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    """Return at most ``max_rows`` indices while preserving every class."""
    if len(labels) <= max_rows:
        return np.arange(len(labels))
    classes, counts = np.unique(labels, return_counts=True)
    if max_rows < len(classes):
        raise ValueError("max_exact_rows must be at least the number of target classes")
    target_counts = np.floor(max_rows * counts / counts.sum()).astype(int)
    target_counts = np.maximum(target_counts, 1)
    while target_counts.sum() > max_rows:
        candidate = int(np.argmax(np.where(target_counts > 1, target_counts, -1)))
        target_counts[candidate] -= 1
    while target_counts.sum() < max_rows:
        candidate = int(np.argmax(counts - target_counts))
        target_counts[candidate] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for class_value, target_count in zip(classes, target_counts):
        class_indices = np.flatnonzero(labels == class_value)
        selected.append(rng.choice(class_indices, size=int(target_count), replace=False))
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result
