"""Train-only preprocessing and feature schema persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ids.data.validate import feature_columns


@dataclass
class FeaturePreprocessor:
    """Numeric imputer/scaler wrapper with a frozen column contract."""

    columns: list[str]
    imputer: Any
    scaler: Any

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Iterable[str] | None = None) -> "FeaturePreprocessor":
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        columns = list(columns or feature_columns(frame))
        if not columns:
            raise ValueError("No numeric predictive features were found")
        values = frame[columns].apply(pd.to_numeric, errors="coerce")
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        scaler.fit(imputer.fit_transform(values))
        return cls(columns=columns, imputer=imputer, scaler=scaler)

    def transform(self, frame: pd.DataFrame):
        values = frame.reindex(columns=self.columns).apply(pd.to_numeric, errors="coerce")
        return self.scaler.transform(self.imputer.transform(values))

    @classmethod
    def fit_transform(cls, frame: pd.DataFrame, columns: Iterable[str] | None = None):
        """Fit a new preprocessor on ``frame`` and return its transformed data.

        ``fit`` remains the API for retaining the fitted object.  This helper
        mirrors the usual transformer contract and, unlike the old instance
        method, cannot accidentally transform with an unfitted object.
        """
        fitted = cls.fit(frame, columns)
        return fitted.transform(frame)

    def to_metadata(self) -> dict:
        return {"columns": self.columns, "feature_count": len(self.columns)}

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePreprocessor":
        import joblib

        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, received {type(loaded).__name__}")
        return loaded
