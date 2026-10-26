"""Leakage-safe out-of-fold stacking for known attack families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StackedClassifier:
    base_models: dict[str, Any]
    meta_model: Any
    classes_: np.ndarray
    model_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        X,
        y,
        *,
        n_splits: int = 5,
        seed: int = 42,
        random_forest_estimators: int = 200,
        xgboost_estimators: int = 200,
        max_rows: int | None = 100000,
    ) -> "StackedClassifier":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        y = np.asarray(y)
        original_classes = np.unique(y)
        if len(original_classes) < 2:
            raise ValueError("Stacking requires at least two attack classes")
        if max_rows is not None and len(y) > max_rows:
            indices = _stratified_sample_indices(y, max_rows, seed)
            X = X[indices]
            y = y[indices]
        # XGBoost's sklearn wrapper requires numeric class labels.  Keep the
        # original attack-family names for the public API, but train every
        # base/meta estimator on stable integer codes.
        original_classes = np.unique(y)
        y_encoded = np.searchsorted(original_classes, y)
        classes = np.arange(len(original_classes))
        base_factories = {
            "random_forest": lambda: RandomForestClassifier(n_estimators=random_forest_estimators, class_weight="balanced", random_state=seed, n_jobs=-1),
            "xgboost": lambda: _make_xgb(xgboost_estimators, seed),
            "logistic_regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        }
        minimum_class_count = int(np.min(np.bincount(y_encoded, minlength=len(classes))))
        effective_splits = min(n_splits, minimum_class_count)
        if effective_splits < 2:
            raise ValueError("OOF stacking requires at least two rows per attack class")
        splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
        oof = np.zeros((len(y), len(base_factories) * len(classes)), dtype=float)
        for model_index, (name, factory) in enumerate(base_factories.items()):
            for train_idx, validation_idx in splitter.split(X, y):
                model = factory()
                model.fit(X[train_idx], y_encoded[train_idx])
                probabilities = _aligned_probabilities(model, X[validation_idx], classes)
                start = model_index * len(classes)
                oof[validation_idx, start:start + len(classes)] = probabilities
        meta_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        meta_model.fit(oof, y_encoded)
        base_models = {name: factory().fit(X, y_encoded) for name, factory in base_factories.items()}
        return cls(base_models=base_models, meta_model=meta_model, classes_=original_classes, model_names=tuple(base_factories))

    def predict_proba(self, X) -> np.ndarray:
        encoded_classes = np.arange(len(self.classes_))
        features = np.hstack([_aligned_probabilities(self.base_models[name], X, encoded_classes) for name in self.model_names])
        return _aligned_probabilities(self.meta_model, features, np.arange(len(self.classes_)))

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_with_confidence(self, X) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.predict_proba(X)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices], probabilities[np.arange(len(probabilities)), indices]


def _make_xgb(n_estimators: int, seed: int):
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, objective="multi:softprob", eval_metric="mlogloss", random_state=seed, n_jobs=1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=min(n_estimators, 200), random_state=seed)


def _aligned_probabilities(model, X, classes: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(X)
    aligned = np.zeros((len(X), len(classes)), dtype=float)
    for index, class_value in enumerate(model.classes_):
        target_index = int(np.where(classes == class_value)[0][0])
        aligned[:, target_index] = probabilities[:, index]
    return aligned


def _stratified_sample_indices(labels: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if len(labels) <= max_rows:
        return np.arange(len(labels))
    classes, counts = np.unique(labels, return_counts=True)
    if max_rows < 2 * len(classes):
        raise ValueError("max_rows must be at least two times the number of target classes")
    target_counts = np.maximum(np.floor(max_rows * counts / counts.sum()).astype(int), 2)
    while target_counts.sum() > max_rows:
        candidate = int(np.argmax(np.where(target_counts > 2, target_counts, -1)))
        if target_counts[candidate] <= 2:
            raise ValueError("max_rows is too small to retain two rows per target class")
        target_counts[candidate] -= 1
    while target_counts.sum() < max_rows:
        candidate = int(np.argmax(counts - target_counts))
        target_counts[candidate] += 1
    rng = np.random.default_rng(seed)
    selected = [
        rng.choice(np.flatnonzero(labels == class_value), size=int(target_count), replace=False)
        for class_value, target_count in zip(classes, target_counts)
    ]
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result
