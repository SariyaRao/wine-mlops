"""Stage 3: evaluate on the held-out test set and log to MLflow."""
import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def start_evaluation_run(run_id: str, experiment_name: str):
    """Resume the training run if it exists, otherwise start a new one.

    The MLflow database is local (not in Git), so on a fresh machine such as
    the CI runner the training run does not exist and we log to a new run.
    """
    try:
        mlflow.get_run(run_id)
        return mlflow.start_run(run_id=run_id)
    except MlflowException:
        mlflow.set_experiment(experiment_name)
        return mlflow.start_run(run_name="evaluate-only")


def main() -> None:
    meta = json.loads(Path("model/metadata.json").read_text())
    params = yaml.safe_load(Path("params.yaml").read_text())
    model = joblib.load("model/model.joblib")

    test = pd.read_csv("data/processed/test.csv")
    X, y = test[meta["features"]], test["target"]
    pred = model.predict(X)

    scores = {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "precision_macro": precision_score(y, pred, average="macro"),
        "recall_macro": recall_score(y, pred, average="macro"),
    }

    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/scores.json").write_text(json.dumps(scores, indent=2))

    Path("reports").mkdir(exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y, pred, cmap="Blues")
    plt.title("Confusion matrix (test set)")
    plt.savefig("reports/confusion_matrix.png", dpi=120, bbox_inches="tight")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    with start_evaluation_run(meta["run_id"], params["mlflow"]["experiment_name"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in scores.items()})
        mlflow.log_artifact("reports/confusion_matrix.png")

    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()