"""Thin Streamlit UI; model execution remains in FastAPI."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

# Streamlit executes this file as a script, which may put ``dashboard/`` on
# sys.path instead of the project root. Make the package import work in both
# ``streamlit run dashboard/app.py`` and normal Python imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.api_client import ApiClient


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Multi-Stage IDS", layout="wide")
    st.title("Multi-Stage ML-Powered Intrusion Detection Engine")
    st.caption("Research prototype: anomaly means unusual relative to benign training traffic, not confirmed zero-day activity.")
    client = ApiClient()
    try:
        health = client.health()
        st.success(f"Backend: {health.get('status')} — {health.get('model_version')}")
    except Exception as exc:
        st.error(f"Backend unavailable: {exc}")
        return
    with st.expander("Model information"):
        st.json(client.model_info())
    uploaded = st.file_uploader("Upload a CICIDS-style CSV", type=["csv"])
    if uploaded:
        content = uploaded.getvalue()
        preview = pd.read_csv(io.BytesIO(content), nrows=10)
        st.dataframe(preview, use_container_width=True)
        if st.button("Predict uploaded flows"):
            try:
                result = client.predict_file(content, uploaded.name)
                predictions = pd.DataFrame(result.get("predictions", []))
                st.dataframe(predictions, use_container_width=True)
                st.download_button("Download predictions", predictions.to_csv(index=False), "predictions.csv", "text/csv")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
