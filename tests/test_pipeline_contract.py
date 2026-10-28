import numpy as np
import pandas as pd
import pytest

from ids.pipeline.predict import InferenceEngine, prediction_columns


class FakePreprocessor:
    columns = ["x"]

    def transform(self, frame):
        return frame[["x"]].to_numpy(float)


class FakeSVM:
    threshold = 0.5

    def score(self, X):
        return X[:, 0]

    def predict(self, X):
        return (X[:, 0] >= self.threshold).astype(int)


class FakeAutoencoder:
    threshold = 0.9

    def score(self, X):
        return X[:, 0]


class FakeStack:
    def __init__(self):
        self.seen_rows = 0

    def predict_with_confidence(self, X):
        self.seen_rows = len(X)
        return np.array(["DoS"] * len(X)), np.array([0.9] * len(X))


def test_prediction_contract_preserves_stage_outputs():
    engine = InferenceEngine(preprocessor=FakePreprocessor(), svm_filter=FakeSVM(), model_version="test")
    result = engine.predict_frame(pd.DataFrame({"flow_id": ["a", "b"], "x": [0.1, 0.9]}))
    assert result.columns.tolist() == prediction_columns()
    assert result["final_alert_label"].tolist() == ["benign", "suspicious_unclassified"]
    assert result["model_version"].tolist() == ["test", "test"]


def test_policy_anomaly_threshold_overrides_model_threshold():
    from ids.pipeline.fuse import FusionPolicy

    engine = InferenceEngine(
        preprocessor=FakePreprocessor(),
        svm_filter=FakeSVM(),
        autoencoder=FakeAutoencoder(),
        policy=FusionPolicy(anomaly_threshold=0.1),
    )
    result = engine.predict_frame(pd.DataFrame({"x": [0.2]}))
    assert bool(result.loc[0, "anomaly_flag"])


class FakeHMM:
    emission_columns = ("svm_score", "known_attack_confidence", "anomaly_score")

    def predict_context(self, observations):
        return {"current_state": len(observations) - 1, "next_state": 1, "next_state_confidence": 0.75}


def test_prediction_applies_temporal_context():
    engine = InferenceEngine(preprocessor=FakePreprocessor(), svm_filter=FakeSVM(), temporal_hmm=FakeHMM())
    result = engine.predict_frame(pd.DataFrame({"x": [0.1, 0.9], "sequence_id": ["a", "a"]}))
    assert result["hmm_current_state"].tolist() == [0, 1]
    assert result["hmm_next_state"].tolist() == [1, 1]


def test_stack_is_only_called_for_svm_gated_rows_and_flow_id_is_preserved():
    stack = FakeStack()
    engine = InferenceEngine(preprocessor=FakePreprocessor(), svm_filter=FakeSVM(), stack_classifier=stack)
    result = engine.predict_frame(pd.DataFrame({"Flow ID": ["first", "second"], "x": [0.1, 0.9]}))
    assert stack.seen_rows == 1
    assert result["flow_id"].tolist() == ["first", "second"]
    assert pd.isna(result.loc[0, "known_attack_label"])
    assert result.loc[1, "known_attack_label"] == "DoS"


def test_inference_rejects_targets_and_nonfinite_features():
    engine = InferenceEngine(preprocessor=FakePreprocessor(), svm_filter=FakeSVM())
    with pytest.raises(ValueError, match="Target/evaluation"):
        engine.predict_frame(pd.DataFrame({"x": [1.0], "Label": ["BENIGN"]}))
    with pytest.raises(ValueError, match="Non-finite"):
        engine.predict_frame(pd.DataFrame({"x": [np.inf]}))


def test_inference_imputes_missing_values_and_accepts_cleaning_metadata():
    engine = InferenceEngine(preprocessor=FakePreprocessor(), svm_filter=FakeSVM())
    result = engine.predict_frame(pd.DataFrame({"x": [np.nan, 0.9], "duplicate_row": [0, 1]}))
    assert len(result) == 2
    with pytest.raises(ValueError, match="non-numeric"):
        engine.predict_frame(pd.DataFrame({"x": ["not-a-number"]}))
