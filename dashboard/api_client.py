"""Small HTTP client for the Streamlit dashboard."""

from __future__ import annotations

import os
from typing import Any

import requests


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 30.0):
        self.base_url = (base_url or os.getenv("IDS_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def model_info(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/model-info", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def predict(self, flow: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/predict", json={"flow": flow}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def predict_file(self, content: bytes, filename: str = "flows.csv") -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/predict-file", files={"file": (filename, content, "text/csv")}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
