"""Stage 2: train a model, track everything in MLflow, save it for DVC."""
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

CLASS_NAMES = ["class_0", "class_1", "class_2"]


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    train_params, mlflow_params = params["train"], params["mlflow"]

    train = pd.read_csv("data/processed/train.csv")
    X, y = train.drop(columns="target"), train["target"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(mlflow_params["experiment_name"])

    register = mlflow_params["register_model"] or os.getenv("REGISTER_MODEL") == "true"

    with mlflow.start_run() as run:
        mlflow.log_params(train_params)
        if sha := os.getenv("GITHUB_SHA"):
            mlflow.set_tag("git_sha", sha)

        model = RandomForestClassifier(**train_params, n_jobs=-1)

        cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        model.fit(X, y)

        mlflow.log_metric("cv_accuracy_mean", cv.mean())
        mlflow.log_metric("cv_accuracy_std", cv.std())
        mlflow.log_metric("train_accuracy", model.score(X, y))

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=infer_signature(X, model.predict(X)),
            input_example=X.head(3),
            registered_model_name=mlflow_params["registered_model_name"] if register else None,
        )

        # Files DVC tracks (the API serves these).
        Path("model").mkdir(exist_ok=True)
        joblib.dump(model, "model/model.joblib")
        metadata = {
            "run_id": run.info.run_id,
            "features": list(X.columns),
            "classes": CLASS_NAMES,
            "params": train_params,
        }
        Path("model/metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"MLflow run: {run.info.run_id}  cv_acc={cv.mean():.3f}")


if __name__ == "__main__":
    main()
