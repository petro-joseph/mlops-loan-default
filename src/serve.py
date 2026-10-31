"""Serve the production model over HTTP, with Prometheus metrics.

Loads `models:/loan-default@production` at start-up, then checks the alias every
RELOAD_SECONDS. When it moves (a promotion or a rollback), the new version is
loaded and swapped in without a restart. If a load fails, the version already
in memory keeps serving and the failure is counted.

    POST /predict   {"loans": [{...20 applicant fields...}]}
    GET  /health    200 with the serving version; 503 until a model is loaded
    GET  /metrics   Prometheus
"""
import asyncio
import os
import time
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

MODEL_NAME = "loan-default"
ALIAS = "production"
RELOAD_SECONDS = int(os.environ.get("RELOAD_SECONDS", "30"))
THRESHOLD = 0.5  # probability of default at or above this -> refer to a credit officer

REQUESTS = Counter("loan_api_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram("loan_api_request_seconds", "Request latency", ["path"])
PREDICTIONS = Counter("loan_predictions_total", "Loans scored", ["decision"])
SCORES = Histogram("loan_default_probability", "Predicted probability of default",
                   buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
VERSION = Gauge("loan_model_version", "Model version being served (0 = none)")
RELOADS = Counter("loan_model_reloads_total", "Model loads", ["result"])

state = {"model": None, "version": None, "columns": []}


def load_production():
    """Load the alias target if it differs from what is in memory."""
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
    if mv.version == state["version"]:
        return
    uri = f"models:/{MODEL_NAME}/{mv.version}"
    model = mlflow.sklearn.load_model(uri)
    columns = mlflow.models.get_model_info(uri).signature.inputs.input_names()
    state.update(model=model, version=mv.version, columns=columns)
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


@asynccontextmanager
async def lifespan(_):
    task = asyncio.create_task(watch_alias())
    yield
    task.cancel()


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
    return {"model_version": version, "results": results}
