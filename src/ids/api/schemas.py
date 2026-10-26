"""Typed API contracts and input validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FlowRequest(BaseModel):
    flow: dict[str, Any]


class BatchRequest(BaseModel):
    flows: list[dict[str, Any]] = Field(min_length=1)


class PredictionResponse(BaseModel):
    prediction: dict[str, Any]
    model_version: str


class BatchResponse(BaseModel):
    predictions: list[dict[str, Any]]
    model_version: str
