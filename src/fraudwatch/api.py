# run: uvicorn fraudwatch.api:app --reload
from __future__ import annotations

import json
import math
import time
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, create_model

from .data import FEATURES, PROJECT_ROOT
from .models import top_reasons
from .policy import BLOCK_THRESHOLD, decide

MODELS = PROJECT_ROOT / "models"
DEMO_FILE = PROJECT_ROOT / "reports" / "demo_transactions.json"
STATIC = Path(__file__).parent / "static" / "index.html"

app = FastAPI(title="FraudWatch", version="1.0")

# v1..v28 are the anonymised PCA features from the dataset
Transaction = create_model(
    "Transaction",
    amount=(float, Field(..., ge=0)),
    hour=(int, Field(..., ge=0, le=23)),
    **{f"v{i}": (float, ...) for i in range(1, 29)},
)


class Reason(BaseModel):
    feature: str
    value: float
    contribution: float


class ScoreResponse(BaseModel):
    fraud_probability: float
    decision: str
    reasons: list[Reason]
    latency_ms: float


@lru_cache(maxsize=1)
def load():
    path = MODELS / "lightgbm.joblib"
    if not path.exists():
        raise HTTPException(503, "model not trained, run python -m fraudwatch.train first")
    thresholds = json.loads((MODELS / "thresholds.json").read_text())
    return joblib.load(path), thresholds


def to_frame(t: dict) -> pd.DataFrame:
    row = {f"V{i}": t[f"v{i}"] for i in range(1, 29)}
    row["log_amount"] = math.log1p(t["amount"])
    row["hour"] = t["hour"]
    return pd.DataFrame([row])[FEATURES]


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return STATIC.read_text(encoding="utf-8")


@app.get("/health")
def health():
    _, thresholds = load()
    return {"status": "ok", "thresholds": {**thresholds, "block": BLOCK_THRESHOLD}}


@app.get("/examples")
def examples():
    # real transactions from the test period
    return json.loads(DEMO_FILE.read_text()) if DEMO_FILE.exists() else []


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction):
    start = time.perf_counter()
    model, thresholds = load()
    X = to_frame(txn.model_dump())
    p = float(model.predict_proba(X)[:, 1][0])
    return {
        "fraud_probability": round(p, 4),
        "decision": decide(p, thresholds["review"]),
        "reasons": top_reasons(model, X)[0],
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }
