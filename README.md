# Fraud Detection MLOps Project

An end-to-end MLOps project for fraud detection with:

- MLflow for experiment tracking and model registry
- Feast for offline and online features
- FastAPI for model serving
- Streamlit for the demo UI
- PostgreSQL for prediction and drift-monitoring storage
- Docker Compose for orchestration

The current workflow is registry-driven: the API serves the model version that is promoted in MLflow with the `champion` alias.

## Project Flow

The project runs in this sequence:

1. Start infrastructure services: PostgreSQL and MLflow.
2. Run the one-time `setup` job.
3. In MLflow, review the trained model versions and assign the `champion` alias to the best one.
4. Start the API and Streamlit services.

That means the application layer does not serve "the latest trained model" automatically. It serves the model version explicitly promoted in the MLflow Model Registry.

## Architecture

```text
Synthetic data -> Feast feature prep -> MLflow training + registry
                    |
                    v
      Promote best version as @champion
                    |
                    v
Postgres <-> FastAPI <-> Feast online features
                    |
                    v
               Streamlit UI
```

## Repository Structure

```text
mlops/
|-- app/
|   |-- config.py                  # Environment-based app configuration
|   |-- data_validation.py         # Request/data validation helpers
|   |-- db.py                      # Postgres persistence for predictions and drift results
|   |-- feast_feature.py           # Feast offline/online feature retrieval
|   |-- generate_data.py           # Synthetic fraud dataset generation
|   |-- monitoring.py              # Drift detection logic
|   |-- prepare_feast_feature.py   # Merchant feature computation + Feast materialization
|   |-- serve_models.py            # FastAPI inference service
|   |-- streamlit_app.py           # Streamlit demo and monitoring dashboard
|   `-- train_mlflow.py            # MLflow training and model registration
|-- data/                          # Generated datasets and Feast artifacts
|-- feature_repo/
|   |-- feature_definitions.py     # Feast entities and feature views
|   `-- feature_store.yaml         # Feast repo configuration
|-- docker-compose.yml             # Service orchestration
|-- Dockerfile.api                 # API image
|-- Dockerfile.mlflow              # MLflow server image
|-- Dockerfile.setup               # One-shot setup/training image
|-- Dockerfile.streamlit           # Streamlit image
|-- pyproject.toml                 # Python project metadata
|-- requirements.txt               # Dependency snapshot
`-- README.md
```

## Services

### `postgres`

Stores:

- API prediction logs
- Drift check history
- MLflow backend metadata

### `mlflow`

Provides:

- experiment tracking
- model registry
- model alias management
- artifact serving

### `setup`

This is a one-shot initialization job. It performs the pipeline in order:

1. cleans stale local Feast artifacts
2. generates training and test data
3. prepares and materializes Feast features
4. trains multiple model runs
5. registers trained models in MLflow
6. saves the label encoder to `models/encoder.pkl`

### `api`

The FastAPI service:

- loads the model from `models:/fraud-detection-model@champion`
- loads the encoder from `models/encoder.pkl`
- fetches live merchant features from Feast
- validates incoming transactions
- applies API key auth and rate limits
- stores predictions in PostgreSQL
- exposes monitoring endpoints

### `streamlit`

The Streamlit app provides:

- transaction scoring UI
- monitoring dashboard
- drift summary view
- recent predictions view

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Optional for local Python runs: Python 3.10 or 3.11 and `uv`

### 1. Start PostgreSQL and MLflow

```bash
docker compose up postgres mlflow -d
```

Once these services are healthy, MLflow is available at [http://localhost:5000](http://localhost:5000).

### 2. Run the setup job

```bash
docker compose run --rm setup
```

This command prepares the full project state:

- creates the dataset
- computes and materializes Feast features
- runs the MLflow training sweep
- registers model versions
- writes `models/encoder.pkl`

### 3. Promote the best model in MLflow

Open [http://localhost:5000](http://localhost:5000), compare the registered model versions, and assign the `champion` alias to the best one.

The API expects:

- registered model name: `fraud-detection-model`
- alias used for serving: `champion`

Important: the API will fail to start correctly if no model version has the `champion` alias.

### 4. Start the API and Streamlit app

```bash
docker compose up api streamlit -d
```

Then open:

- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Streamlit UI: [http://localhost:8501](http://localhost:8501)

## Runtime Configuration

The API and Streamlit services use `.env.api`.

Important values:

- `FRAUD_API_KEY`: required for protected API endpoints
- `MLFLOW_MODEL_NAME`: defaults to `fraud-detection-model`
- `MLFLOW_MODEL_ALIAS`: defaults to `champion`
- `FRAUD_THRESHOLD`: default threshold, overridden by MLflow if the selected run logged `optimal_threshold`
- `REFERENCE_DATA_PATH`: reference dataset used for drift monitoring

Example header for protected endpoints:

```http
x-api-key: supersecretapikey
```

## API Overview

### Main endpoints

- `POST /predict` - score a transaction
- `GET /health` - service health check
- `GET /model-info` - loaded model metadata
- `GET /monitoring/stats` - aggregate prediction metrics
- `GET /monitoring/recent` - latest predictions
- `GET /monitoring/drift` - drift summary and history
- `GET /monitoring/drift/check?window=100` - run a manual drift check

### Example prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: supersecretapikey" \
  -d '{
    "amount": 850.0,
    "hour": 2,
    "day_of_week": 6,
    "merchant_category": "online"
  }'
```

## Local Development

If you want to run parts of the project without Docker for the app layer:

```bash
uv sync
uv run uvicorn app.serve_models:app --host 0.0.0.0 --port 8000
uv run streamlit run app/streamlit_app.py
```

For local runs, make sure:

- PostgreSQL is available
- MLflow is running
- the setup flow has already been completed
- a `champion` model alias exists in MLflow
- `models/encoder.pkl` exists

## Operational Notes

- `setup` is intended as a batch initialization step, not a long-running service.
- Feast online lookups have a fallback path if a merchant feature is unavailable.
- Predictions and drift checks are persisted in PostgreSQL.
- Drift monitoring uses `data/train.csv` as the default reference dataset.
- The API reads the promoted model version at startup, so if you change the `champion` alias, restart the API to load the new model.

## Recommended Startup Commands

```bash
docker compose up postgres mlflow -d
docker compose run --rm setup
# Promote best registered model as @champion in MLflow UI
docker compose up api streamlit -d
```

## Tech Stack

- Python
- FastAPI
- Streamlit
- MLflow
- Feast
- PostgreSQL
- scikit-learn
- Pandas
- Docker Compose

## License

This repository is intended for learning, experimentation, and portfolio use.
