from typing import Any, cast

import pandas as pd
from fastapi.testclient import TestClient

from ids.api.main import create_app
from ids.api.service import InferenceService


class _FakeEngine:
    preprocessor = object()
    svm_filter = object()
    model_version = "test"
    required_features = []
    stack_classifier = None

    def predict_frame(self, frame):
        return pd.DataFrame({"flow_id": ["uploaded-flow"], "svm_score": [0.1]})


def test_predict_file_accepts_csv_upload():
    client = TestClient(create_app(InferenceService(cast(Any, _FakeEngine()))))

    response = client.post(
        "/predict-file",
        files={"file": ("flows.csv", b"feature\n1\n", "text/csv")},
    )

    assert response.status_code == 200
    assert response.json()["predictions"] == [{"flow_id": "uploaded-flow", "svm_score": 0.1}]


def test_predict_file_enforces_batch_limit():
    client = TestClient(create_app(InferenceService(cast(Any, _FakeEngine()), max_batch_rows=1)))
    response = client.post(
        "/predict-file",
        files={"file": ("flows.csv", b"feature\n1\n2\n", "text/csv")},
    )
    assert response.status_code == 413


def test_predict_file_rejects_empty_csv():
    client = TestClient(create_app(InferenceService(cast(Any, _FakeEngine()))))
    response = client.post(
        "/predict-file",
        files={"file": ("flows.csv", b"feature\n", "text/csv")},
    )
    assert response.status_code == 422


class _PartlyGatedEngine(_FakeEngine):
    """Mimics real output: confidence is missing (NaN) for ungated rows."""

    def predict_frame(self, frame):
        confidences = [0.9 if index % 2 else None for index in range(len(frame))]
        return pd.DataFrame({"flow_id": [str(index) for index in range(len(frame))], "known_attack_confidence": confidences})


def test_predict_endpoints_accept_json_bodies_and_serialize_missing_values():
    client = TestClient(create_app(InferenceService(cast(Any, _PartlyGatedEngine()))))

    single = client.post("/predict", json={"flow": {"feature": 1}})
    batch = client.post("/predict-batch", json={"flows": [{"feature": 1}, {"feature": 2}]})
    upload = client.post("/predict-file", files={"file": ("flows.csv", b"feature\n1\n2\n", "text/csv")})

    assert single.status_code == 200
    assert single.json()["prediction"]["known_attack_confidence"] is None
    assert batch.status_code == 200
    assert [row["known_attack_confidence"] for row in batch.json()["predictions"]] == [None, 0.9]
    assert upload.status_code == 200
    assert [row["known_attack_confidence"] for row in upload.json()["predictions"]] == [None, 0.9]


def test_openapi_schema_builds():
    client = TestClient(create_app(InferenceService(cast(Any, _FakeEngine()))))
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "requestBody" in response.json()["paths"]["/predict"]["post"]
