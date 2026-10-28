"""Causal temporal modeling with hmmlearn when available and a transparent fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


def build_sequences(frame: pd.DataFrame, *, sequence_column: str = "sequence_id", timestamp_column: str = "timestamp", min_length: int = 2, emission_columns: Iterable[str] = ()) -> list[np.ndarray]:
    if sequence_column not in frame.columns:
        return []
    emission_columns = list(emission_columns)
    available = [column for column in emission_columns if column in frame.columns]
    if not available:
        raise ValueError("No configured HMM emission columns are present")
    working = frame.copy()
    if timestamp_column in working:
        working["__timestamp"] = pd.to_datetime(working[timestamp_column], errors="coerce")
    else:
        working["__timestamp"] = pd.NaT
    sequences = []
    for _, group in working.dropna(subset=[sequence_column]).groupby(sequence_column, sort=False):
        ordered = group.sort_values("__timestamp", kind="stable") if group["__timestamp"].notna().any() else group
        if len(ordered) >= min_length:
            sequences.append(ordered[available].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float))
    return sequences


@dataclass
class TemporalHMM:
    model: Any
    state_count: int
    emission_columns: tuple[str, ...]
    fallback_transition: np.ndarray | None = None

    @classmethod
    def fit(cls, sequences: list[np.ndarray], *, state_count: int = 3, emission_columns: Iterable[str] = ()) -> "TemporalHMM":
        if not sequences:
            raise ValueError("At least one sequence of length >= 2 is required")
        try:
            from hmmlearn.hmm import GaussianHMM

            model = GaussianHMM(n_components=state_count, covariance_type="diag", n_iter=100, random_state=42)
            model.fit(np.vstack(sequences), lengths=[len(sequence) for sequence in sequences])
            return cls(model=model, state_count=state_count, emission_columns=tuple(emission_columns))
        except ImportError:
            values = np.vstack(sequences)
            states = np.argmin(np.linalg.norm(values[:, None, :] - np.quantile(values, np.linspace(0, 1, state_count), axis=0)[None, :, :], axis=2), axis=1)
            transition = np.ones((state_count, state_count), dtype=float) * 1e-6
            offset = 0
            for sequence in sequences:
                sequence_states = states[offset:offset + len(sequence)]
                for current, next_state in zip(sequence_states, sequence_states[1:]):
                    transition[current, next_state] += 1
                offset += len(sequence)
            transition /= transition.sum(axis=1, keepdims=True)
            return cls(model=None, state_count=state_count, emission_columns=tuple(emission_columns), fallback_transition=transition)

    def current_state(self, observations: np.ndarray) -> int:
        observations = np.asarray(observations, dtype=float)
        if self.model is not None:
            # ``predict`` on only the last row discards the temporal history.
            # Filtering the observations seen so far keeps this causal while
            # allowing earlier emissions to influence the current state.
            posterior = np.asarray(self.model.predict_proba(observations), dtype=float)
            return int(np.argmax(posterior[-1]))
        return int(np.argmax(observations[-1])) % self.state_count

    def next_state_distribution(self, observations: np.ndarray) -> np.ndarray:
        state = self.current_state(observations)
        if self.model is not None:
            posterior = np.asarray(self.model.predict_proba(observations), dtype=float)[-1]
            distribution = posterior @ np.asarray(self.model.transmat_, dtype=float)
            total = distribution.sum()
            return distribution / total if total else np.full(self.state_count, 1.0 / self.state_count)
        if self.fallback_transition is None:
            raise RuntimeError("Temporal model has no transition matrix")
        return np.asarray(self.fallback_transition[state], dtype=float)

    def predict_context(self, observations: np.ndarray) -> dict:
        distribution = self.next_state_distribution(observations)
        next_state = int(np.argmax(distribution))
        return {"current_state": self.current_state(observations), "next_state": next_state, "next_state_confidence": float(distribution[next_state])}
