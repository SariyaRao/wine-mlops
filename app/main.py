"""FastAPI service that serves the DVC-versioned model."""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import BatchRequest, BatchResponse, Prediction, WineFeatures

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", "model/model.joblib"))
    meta_path = Path(os.getenv("METADATA_PATH", "model/metadata.json"))
    if not model_path.exists():
        raise RuntimeError(f"{model_path} not found. Run `dvc pull` or `dvc repro` first.")
    state["model"] = joblib.load(model_path)
    state["meta"] = json.loads(meta_path.read_text())
    yield
    state.clear()


app = FastAPI(title="Wine Classifier API", version="1.0.0", lifespan=lifespan)


def _predict(rows: list[WineFeatures]) -> list[Prediction]:
    meta = state["meta"]
    df = pd.DataFrame([r.model_dump() for r in rows])[meta["features"]]
    model = state["model"]
    preds = model.predict(df)
    probas = model.predict_proba(df)
    classes = [int(c) for c in model.classes_]
    names = meta["classes"]
    return [
        Prediction(
            prediction=int(p),
            label=names[int(p)],
            probabilities={names[c]: round(float(pr), 4) for c, pr in zip(classes, row)},
        )
        for p, row in zip(preds, probas)
    ]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    meta = state["meta"]
    return {"mlflow_run_id": meta["run_id"], "features": meta["features"],
            "classes": meta["classes"], "params": meta["params"]}


@app.post("/predict", response_model=Prediction)
def predict(features: WineFeatures):
    try:
        return _predict([features])[0]
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    return BatchResponse(predictions=_predict(req.instances))
