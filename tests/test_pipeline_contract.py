import numpy as np
import pandas as pd

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
