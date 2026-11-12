# Wine Classifier — an end-to-end MLOps project

A small, complete project that wires together four tools:

| Tool | Role in this project |
|---|---|
| **DVC + AWS S3** | Versions the dataset and model; `dvc.yaml` defines the pipeline |
| **MLflow** | Tracks params, metrics, artifacts; logs the model; optional registry |
| **GitHub Actions** | CI (lint, tests, reproduce pipeline, quality gate) and CD (Docker image) |
| **FastAPI** | Serves the model over HTTP |

```
raw CSV ──dvc add──▶ prepare ──▶ train ──▶ evaluate ──▶ metrics/scores.json
 (S3)                  │          │  └─ MLflow run (params, metrics, model)
                       ▼          ▼
                 train/test    model/model.joblib (S3) ──▶ FastAPI ──▶ Docker image (GHCR)
```

## Project layout

```
.github/workflows/ci-cd.yml   GitHub Actions
app/                          FastAPI service (main.py, schemas.py)
src/                          pipeline stages: prepare.py, train.py, evaluate.py
scripts/                      make_dataset.py, check_metrics.py, setup_dvc_s3.sh
tests/                        API tests (self-contained, no S3 needed)
dvc.yaml  params.yaml         pipeline + hyperparameters
Dockerfile  Makefile
```

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
git init && git add . && git commit -m "Initial project"
```

## 2. Create the S3 remote (AWS)

1. Create a private S3 bucket, e.g. `my-mlops-bucket` (console → S3 → Create bucket).
2. Create an IAM user with this policy (replace the bucket name), then create an access key:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
     "Resource": "arn:aws:s3:::my-mlops-bucket"},
    {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
     "Resource": "arn:aws:s3:::my-mlops-bucket/*"}
  ]
}
```

3. Give your terminal the credentials (never commit them):

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
```

4. Configure DVC:

```bash
./scripts/setup_dvc_s3.sh my-mlops-bucket us-east-1
```

## 3. Version the data

```bash
make data                       # writes data/raw/wine.csv
dvc add data/raw/wine.csv       # creates data/raw/wine.csv.dvc and git-ignores the CSV
git add data/raw/wine.csv.dvc data/raw/.gitignore .dvc
git commit -m "Track raw data with DVC"
```

## 4. Run the pipeline

```bash
dvc repro                       # prepare -> train -> evaluate
dvc metrics show                # accuracy, f1, ...
```

Run it again: DVC says nothing changed and skips everything. Now edit `params.yaml`
(e.g. `n_estimators: 50`) and run `dvc repro` — only the stages affected re-run.
Compare experiments:

```bash
dvc params diff
dvc metrics diff
```

## 5. Look at MLflow

```bash
make mlflow-ui                  # open http://localhost:5000
```

Each `dvc repro` of `train` creates a run with parameters, CV metrics, the logged
model (with signature + input example); `evaluate` adds the `test_*` metrics and the
confusion-matrix image to the same run.

**Model Registry (optional):** set `register_model: true` in `params.yaml`
(or `REGISTER_MODEL=true dvc repro`) and the model appears under the *Models* tab as
`wine-classifier`, with a new version per run. The registry needs a database-backed
store, which is why the default tracking URI is `sqlite:///mlflow.db`. For a team, run
a shared MLflow server and set `MLFLOW_TRACKING_URI` to its URL.

## 6. Push versions to S3 and commit

```bash
dvc push                                   # data + model -> S3
git add dvc.lock metrics/scores.json params.yaml .gitignore
git commit -m "Train model v1"
```

`dvc.lock` is the key file: it records the exact hashes of data, params and model, so any
teammate (or CI) can `git checkout <commit> && dvc pull` to get that exact version.

## 7. Serve with FastAPI

```bash
make serve                                 # http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "alcohol": 13.2, "malic_acid": 2.77, "ash": 2.51, "alcalinity_of_ash": 18.5,
  "magnesium": 96, "total_phenols": 1.9, "flavanoids": 0.58, "nonflavanoid_phenols": 0.5,
  "proanthocyanins": 1.25, "color_intensity": 5.4, "hue": 0.75, "od280_od315": 1.8,
  "proline": 600}'
```

Endpoints: `GET /health`, `GET /model-info`, `POST /predict`, `POST /predict/batch`.
Docker: `make docker` (needs `model/` present, i.e. after `dvc repro` or `dvc pull`).

## 8. CI/CD with GitHub Actions

1. Push the repo to GitHub.
2. Repo → Settings → Secrets and variables → Actions → add
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`.
3. Open a PR or push to `main`. The workflow runs three jobs:

| Job | Runs on | What it does |
|---|---|---|
| `test` | PRs + main | `ruff` lint, `pytest` API tests |
| `pipeline` | PRs + main | `dvc pull`, `dvc repro`, accuracy gate (≥ 0.90), metrics diff vs. `main`; `dvc push` on main |
| `build-and-push` | main only | `dvc pull` the model, build Docker image, push to `ghcr.io/<you>/<repo>/wine-api` |

Run the container from CI: `docker run -p 8000:8000 ghcr.io/<you>/<repo>/wine-api:latest`.

## Typical day-to-day workflow

```bash
git checkout -b try-deeper-trees
# edit params.yaml
dvc repro && dvc metrics diff
dvc push
git add -A && git commit -m "Deeper trees" && git push   # open a PR; CI re-checks
```

## Troubleshooting

- **`dvc pull` fails in CI** — you forgot `dvc push` locally, or the secrets/region are wrong.
- **Model won't load in the API** — scikit-learn version differs between training and serving; keep both requirements files in sync.
- **`mlflow.db` is not in Git** — intentional; for shared tracking use a hosted MLflow server.
- **Raise/lower the quality gate** — edit the number in `ci-cd.yml` (`check_metrics.py 0.90`).

## Where to go next
Swap the dataset for your own, add data validation (e.g. Great Expectations) as a DVC stage,
deploy the GHCR image to ECS/Cloud Run, and add monitoring/drift checks.
