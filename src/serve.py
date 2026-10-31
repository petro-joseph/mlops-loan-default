"""Serve the production model over HTTP, with Prometheus metrics.

Loads `models:/loan-default@production` at start-up, then checks the alias every
RELOAD_SECONDS. When it moves (a promotion or a rollback), the new version is
loaded and swapped in without a restart. If a load fails, the version already
in memory keeps serving and the failure is counted.

Drift: the last DRIFT_WINDOW applications and scores are kept in memory, and
every DRIFT_SECONDS each feature's PSI against the serving version's training
profile (`reference.json`, logged by train.py) is published as a metric.
The window is cleared when the version changes.

    POST /predict   {"loans": [{...20 applicant fields...}]}
    GET  /health    200 with the serving version; 503 until a model is loaded
    GET  /metrics   Prometheus
"""
import asyncio
import json
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from drift import psi

MODEL_NAME = "loan-default"
ALIAS = "production"
RELOAD_SECONDS = int(os.environ.get("RELOAD_SECONDS", "30"))
THRESHOLD = 0.5  # probability of default at or above this -> refer to a credit officer
DRIFT_WINDOW = int(os.environ.get("DRIFT_WINDOW", "500"))
DRIFT_MIN = 100  # fewer recent applications than this is too noisy to judge
DRIFT_SECONDS = 30

REQUESTS = Counter("loan_api_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram("loan_api_request_seconds", "Request latency", ["path"])
PREDICTIONS = Counter("loan_predictions_total", "Loans scored", ["decision"])
SCORES = Histogram("loan_default_probability", "Predicted probability of default",
                   buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
VERSION = Gauge("loan_model_version", "Model version being served (0 = none)")
RELOADS = Counter("loan_model_reloads_total", "Model loads", ["result"])
FEATURE_PSI = Gauge("loan_feature_psi", "PSI of recent inputs vs training", ["feature"])
SCORE_PSI = Gauge("loan_score_psi", "PSI of recent scores vs validation scores")
WINDOW_SIZE = Gauge("loan_drift_window_size", "Applications in the drift window")

state = {"model": None, "version": None, "columns": [], "reference": None}
window = deque(maxlen=DRIFT_WINDOW)  # (application dict, score)
window_lock = threading.Lock()


def load_production():
    """Load the alias target if it differs from what is in memory."""
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
    if mv.version == state["version"]:
        return
    uri = f"models:/{MODEL_NAME}/{mv.version}"
    model = mlflow.sklearn.load_model(uri)
    columns = mlflow.models.get_model_info(uri).signature.inputs.input_names()
    try:
        with open(mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="reference.json")) as f:
            reference = json.load(f)
    except Exception:
        reference = None  # trained before drift profiles existed: serve, but no drift metrics
        print(f"v{mv.version} has no reference.json; drift metrics off", flush=True)
    with window_lock:
        window.clear()
    FEATURE_PSI.clear()
    SCORE_PSI.set(0)
    state.update(model=model, version=mv.version, columns=columns, reference=reference)
    VERSION.set(int(mv.version))
    RELOADS.labels("ok").inc()
    print(f"serving {MODEL_NAME} v{mv.version}", flush=True)


async def watch_alias():
    while True:
        try:
            await asyncio.to_thread(load_production)
        except Exception as e:  # keep serving the old version
            RELOADS.labels("failed").inc()
            print(f"reload failed, still serving v{state['version']}: {e}", flush=True)
        await asyncio.sleep(RELOAD_SECONDS)


def measure_drift():
    ref = state["reference"]
    with window_lock:
        recent = list(window)
    WINDOW_SIZE.set(len(recent))
    if ref is None or len(recent) < DRIFT_MIN:
        return
    rows = pd.DataFrame([r for r, _ in recent])
    for col, prof in ref["features"].items():
        FEATURE_PSI.labels(col).set(psi(prof, rows[col].tolist()))
    SCORE_PSI.set(psi(ref["score"], [p for _, p in recent]))


async def watch_drift():
    while True:
        await asyncio.sleep(DRIFT_SECONDS)
        try:
            await asyncio.to_thread(measure_drift)
        except Exception as e:
            print(f"drift check failed: {e}", flush=True)


@asynccontextmanager
async def lifespan(_):
    tasks = [asyncio.create_task(watch_alias()), asyncio.create_task(watch_drift())]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="loan-default", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def measure(request: Request, call_next):
    path = request.url.path
    if path.startswith("/metrics"):
        return await call_next(request)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        LATENCY.labels(path).observe(time.perf_counter() - start)
        REQUESTS.labels(path, str(status)).inc()


class ScoreRequest(BaseModel):
    loans: list[dict] = Field(min_length=1, max_length=500)


@app.get("/health")
def health():
    if state["model"] is None:
        return JSONResponse({"status": "no model loaded"}, status_code=503)
    return {"status": "ok", "model": MODEL_NAME, "version": state["version"]}


@app.post("/predict")
def predict(req: ScoreRequest):
    model, version = state["model"], state["version"]  # one consistent pair per request
    if model is None:
        raise HTTPException(503, "no model loaded")
    df = pd.DataFrame(req.loans)
    missing = [c for c in state["columns"] if c not in df.columns]
    if missing:
        raise HTTPException(422, f"missing fields: {missing}")
    try:
        proba = model.predict_proba(df[state["columns"]])[:, 1]
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"invalid input: {e}")
    results = []
    for p in proba:
        decision = "refer" if p >= THRESHOLD else "approve"
        SCORES.observe(p)
        PREDICTIONS.labels(decision).inc()
        results.append({"default_probability": round(float(p), 4), "decision": decision})
    with window_lock:
        window.extend(zip(df[state["columns"]].to_dict("records"), proba.tolist()))
    return {"model_version": version, "results": results}
