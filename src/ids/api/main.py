"""Local FastAPI application for the saved inference engine."""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd

FastAPI: Any = None
File: Any = None
HTTPException: Any = None
UploadFile: Any = None
try:
    from fastapi import FastAPI as _FastAPI
    from fastapi import File as _File
    from fastapi import HTTPException as _HTTPException
    from fastapi import UploadFile as _UploadFile

    FastAPI = _FastAPI
    File = _File
    HTTPException = _HTTPException
    UploadFile = _UploadFile
except ImportError:  # pragma: no cover - optional during lightweight imports
    pass

from ids.api.service import InferenceService
from ids.pipeline.load import PROJECT_ROOT, load_fusion_policy, load_inference_engine
from ids.pipeline.predict import InferenceEngine
from ids.utils.io import read_yaml


def _load_engine() -> InferenceEngine:
    """Load the artifacts produced by the training pipeline.

    The API and Streamlit dashboard must use the same saved models as the CLI
    batch predictor.  Keep paths fixed to the local artifact directory rather
    than accepting model paths from HTTP requests.
    """
    artifacts_dir = Path(os.getenv("IDS_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
    return load_inference_engine(artifacts_dir=artifacts_dir, project_root=PROJECT_ROOT)


def create_app(service: InferenceService | None = None):
    if FastAPI is None or File is None or HTTPException is None or UploadFile is None:  # pragma: no cover
        raise RuntimeError("FastAPI is required to create the HTTP application")
    from ids.api.schemas import BatchRequest, FlowRequest

    api_config = read_yaml(PROJECT_ROOT / "configs" / "api.yaml")
    load_error = None
    if service is None:
        try:
            engine = _load_engine()
        except (ImportError, OSError, RuntimeError) as exc:
            # Keep liveness available while accurately reporting that the
            # server cannot serve predictions until a valid bundle exists.
            engine = InferenceEngine(policy=load_fusion_policy(PROJECT_ROOT), model_version="untrained-development")
            load_error = str(exc)
        service = InferenceService(engine, max_batch_rows=int(api_config.get("max_batch_rows", 10000)))
    max_upload_bytes = int(api_config.get("max_upload_bytes", 10 * 1024 * 1024))

    @asynccontextmanager
    async def lifespan(_app):
        yield

    app = FastAPI(title="Multi-Stage IDS API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        ready = bool(service.engine.preprocessor and service.engine.svm_filter)
        response = {"status": "ready" if ready and load_error is None else "not_ready", "model_version": service.engine.model_version}
        if load_error is not None:
            response["error"] = load_error
        return response

    @app.get("/model-info")
    def model_info():
        return service.model_info

    @app.post("/predict")
    def predict(request: FlowRequest):
        try:
            return {"prediction": service.predict_flow(request.flow), "model_version": service.engine.model_version}
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/predict-batch")
    def predict_batch(request: BatchRequest):
        try:
            return {"predictions": service.predict_flows(request.flows), "model_version": service.engine.model_version}
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/predict-file")
    async def predict_file(file: Any = File(...)):
        if not file.filename or Path(file.filename).suffix.lower() != ".csv":
            raise HTTPException(status_code=415, detail="Only CSV uploads are supported")
        content = await file.read()
        if len(content) > max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"Uploaded file exceeds the {max_upload_bytes}-byte limit")
        try:
            frame = pd.read_csv(io.BytesIO(content))
            if frame.empty:
                raise HTTPException(status_code=422, detail="Uploaded CSV contains no data rows")
            if len(frame) > service.max_batch_rows:
                raise HTTPException(status_code=413, detail=f"Uploaded file exceeds maximum of {service.max_batch_rows} rows")
            predictions_frame = service.engine.predict_frame(frame)
            predictions = predictions_frame.where(pd.notna(predictions_frame), None).to_dict(orient="records")
            return {"predictions": predictions, "model_version": service.engine.model_version}
        except (ValueError, RuntimeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = None
try:
    app = create_app()
except RuntimeError:
    # Importing the package remains possible before optional serving dependencies are installed.
    app = None
