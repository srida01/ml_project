import pandas as pd

from ids.models.temporal_hmm import build_sequences


def test_sequences_are_sorted_and_short_sequences_are_excluded():
    frame = pd.DataFrame({"sequence_id": ["a", "a", "b"], "timestamp": ["2026-01-01 00:02", "2026-01-01 00:01", "2026-01-01"], "svm_score": [2.0, 1.0, 3.0]})
    sequences = build_sequences(frame, emission_columns=["svm_score"])
    assert len(sequences) == 1
    assert sequences[0].ravel().tolist() == [1.0, 2.0]
