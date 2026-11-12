def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict(client, sample_row):
    r = client.post("/predict", json=sample_row)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1, 2)
    assert abs(sum(body["probabilities"].values()) - 1) < 0.01


def test_predict_batch(client, sample_row):
    r = client.post("/predict/batch", json={"instances": [sample_row, sample_row]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_validation_error(client):
    r = client.post("/predict", json={"alcohol": "not-a-number"})
    assert r.status_code == 422


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert "features" in r.json()
