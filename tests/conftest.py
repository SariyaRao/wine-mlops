"""Tests build their own tiny model, so they don't depend on DVC or S3."""
import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture(scope="session")
def sample_row():
    X = load_wine(as_frame=True).data.rename(
        columns={"od280/od315_of_diluted_wines": "od280_od315"}
    )
    return X.iloc[0].to_dict()


@pytest.fixture(scope="session")
def client(tmp_path_factory, monkeypatch_session):
    d = tmp_path_factory.mktemp("model")
    ds = load_wine(as_frame=True)
    X = ds.data.rename(columns={"od280/od315_of_diluted_wines": "od280_od315"})
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, ds.target)
    joblib.dump(model, d / "model.joblib")
    (d / "metadata.json").write_text(json.dumps({
        "run_id": "test-run", "features": list(X.columns),
        "classes": ["class_0", "class_1", "class_2"], "params": {},
    }))
    monkeypatch_session.setenv("MODEL_PATH", str(d / "model.joblib"))
    monkeypatch_session.setenv("METADATA_PATH", str(d / "metadata.json"))

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()
