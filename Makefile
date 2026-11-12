.PHONY: install data pipeline test lint serve mlflow-ui docker

install:
	pip install -r requirements.txt

data:            ## generate the raw CSV (run once, then `dvc add`)
	python scripts/make_dataset.py

pipeline:        ## run the DVC pipeline (prepare -> train -> evaluate)
	dvc repro

test:
	pytest -q

lint:
	ruff check .

serve:
	uvicorn app.main:app --reload --port 8000

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker:
	docker build -t wine-api . && docker run --rm -p 8000:8000 wine-api
