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
