# Fraud Detection MLOps Pipeline

A production-grade, end-to-end MLOps pipeline for real-time credit card fraud detection — built with industry-standard tools to demonstrate the full lifecycle of a machine learning system, from data generation to model monitoring.

---

## Architecture

```
┌──────────────┐     ┌────────────────────┐     ┌───────────────────┐
│  Synthetic   │────▶│  Data Validation  │────▶│   Model Training  │
│  Data Gen    │     │  (Great Expects.)  │     │   (scikit-learn)  │
└──────────────┘     └────────────────────┘     └────────┬──────────┘
                                                         │
                     ┌──────────────────┐                │
                     │  Feature Store   │◀────────────┤
                     │  (Feast)         │              │
                     └──────┬───────────┘              ▼
                            │               ┌─────────────────┐
                            │               │  Experimen      │
                            │               │  Tracking       │
                            │               │  (MLflow)       │
                            │               └────────┬────────┘
                            │                        │
                            ▼                        ▼
                     ┌──────────────────┐    ┌─────────────────┐
                     │  Online Features  │──▶│  Model Serving  │
                     │  (low-latency)    │    │  (FastAPI)       │
                     └──────────────────┘     └────────┬────────┘
                                                       │
                     ┌──────────────────┐              │
                     │  Drift Detection  │◀────────────┘
                     │  (Evidently AI)   │
                     └──────────────────┘
                            │
                     ┌──────────────────┐
                     │  CI/CD Pipeline   │
                     │  (GitHub Actions) │
                     └──────────────────┘
```

---

## Key Features

| Component               | Tool                    | What It Does                                                                     |
| ----------------------- | ----------------------- | -------------------------------------------------------------------------------- |
| **Data Generation**     | NumPy / Pandas          | Generates realistic synthetic transaction data with configurable fraud ratios    |
| **Data Validation**     | Great Expectations      | Validates individual transactions and entire batches against quality rules       |
| **Feature Store**       | Feast                   | Manages merchant-level features for both training (offline) and serving (online) |
| **Experiment Tracking** | MLflow                  | Tracks hyperparameters, metrics, artifacts, and registers models                 |
| **Model Serving**       | FastAPI                 | Production-ready REST API with input validation, health checks, and Swagger docs |
| **Drift Monitoring**    | Evidently AI + SciPy    | Detects data drift using KS tests and generates visual HTML reports              |
| **Containerization**    | Docker / Docker Compose | Packages the API alongside the MLflow server for portable deployment             |
| **CI/CD**               | GitHub Actions          | Automated testing, Docker build, and API validation on every push                |

---

## Project Structure

```
mlops/
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI/CD pipeline definition
├── data/
│   ├── train.csv                   # Training dataset (80%)
│   ├── test.csv                    # Test dataset (20%)
│   └── merchant_features.parquet   # Feast feature source
├── feature_repo/
│   ├── feature_store.yaml          # Feast configuration
│   └── feature_definitions.py      # Entity & FeatureView definitions
├── models/
│   └── model.pkl                   # Trained model artifact
├── src/
│   ├── generate_data.py            # Synthetic data generator
│   ├── data_validation.py          # GX-based validation logic
│   ├── train_naive.py              # Standalone training script
│   ├── train_mlflow.py             # MLflow-tracked training + sweep
│   ├── serve_naive.py              # Basic FastAPI serving (v1)
│   ├── serve_validated.py          # Validated serving with MLflow (v3)
│   ├── serve_mlflow.py             # Champion model serving (v2)
│   ├── monitoring.py               # Drift detection & reporting
│   ├── feast_feature.py            # Feast retrieval helpers
│   ├── prepare_feast_feature.py    # Compute & materialize features
│   └── test_bad_data.py            # Validation edge-case demos
├── tests/
│   ├── test_data_models.py         # Data quality & model performance tests
│   └── test_api.py                 # API integration tests
├── Dockerfile                      # Container image definition
├── docker-compose.yml              # Multi-service orchestration
├── pyproject.toml                  # Project metadata & dependencies
├── requirements.txt                # Pip dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python** 3.10 – 3.11
- **Docker** & **Docker Compose** (for containerized deployment)
- **uv** (recommended) or **pip** for dependency management

### 1. Clone the Repository

```bash
git clone https://github.com/divakar166/mlops.git
cd mlops
```

### 2. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 3. Generate the Dataset

```bash
python src/generate_data.py
```

This creates `data/train.csv` (8,000 transactions) and `data/test.csv` (2,000 transactions) with a 2% fraud ratio. Legitimate transactions follow realistic log-normal spending patterns; fraudulent ones exhibit late-night, high-amount, online-heavy behavior.

### 4. Train the Model

**Option A: Standalone (no MLflow required)**

```bash
python src/train_naive.py
```

Trains a Random Forest classifier and saves `models/model.pkl`.

**Option B: With MLflow tracking + hyperparameter sweep**

```bash
# Start MLflow server first
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# In a new terminal, run the experiment sweep
python src/train_mlflow.py
```

This trains 5 model configurations, logging all params, metrics, and artifacts to MLflow. Visit `http://localhost:5000` to compare runs and promote the best model.

---

## Component Deep Dives

### Data Validation (Great Expectations)

Validates data at **two levels**:

1. **Single-transaction validation** — Real-time checks on individual API requests (amount > 0, valid hour, known merchant category, etc.)
2. **Batch validation** — Full-dataset quality gates using Great Expectations expectation suites (null checks, range constraints, set membership)

```bash
python src/data_validation.py
```

### Feature Store (Feast)

Manages merchant-level aggregate features (average amount, transaction count, fraud rate):

```bash
# Step 1: Compute and materialize features
python src/prepare_feast_feature.py

# Step 2: Test online + offline retrieval
python src/feast_feature.py
```

- **Offline store** — Used during training for point-in-time correct feature joins
- **Online store** — Used during serving for low-latency feature lookups

### Drift Monitoring (Evidently AI)

Simulates four real-world drift scenarios and generates an interactive HTML report:

```bash
python src/monitoring.py
```

| Scenario         | What Changes                | Expected Drift |
| ---------------- | --------------------------- | -------------- |
| Test data        | Nothing (baseline)          | Minimal        |
| Fraud spike      | Fraud rate 2% → 10%         | Moderate       |
| Amount inflation | All amounts ×2              | Amount column  |
| Time shift       | All transactions late-night | Hour column    |

The generated `drift_report.html` can be opened in any browser for detailed visualizations.

### Model Serving (FastAPI)

Three API versions demonstrate progressive improvements:

| Version | Script               | Features                                         |
| ------- | -------------------- | ------------------------------------------------ |
| **v1**  | `serve_naive.py`     | Basic prediction from pickle                     |
| **v2**  | `serve_mlflow.py`    | Loads `@champion` model from MLflow Registry     |
| **v3**  | `serve_validated.py` | v2 + input validation, rejects bad data with 400 |

```bash
# Start the API (v3 - recommended)
uvicorn src.serve_validated:app --host 0.0.0.0 --port 8000

# Interactive docs available at:
# http://localhost:8000/docs
```

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 500.00,
    "hour": 3,
    "day_of_week": 1,
    "merchant_category": "online"
  }'
```

**Example response:**

```json
{
  "is_fraud": true,
  "fraud_probability": 0.87,
  "model_source": "MLflow Production",
  "validation_passed": true
}
```

---

## Docker Deployment

Run the full stack (MLflow server + Fraud Detection API) with a single command:

```bash
docker-compose up --build
```

| Service      | URL                          | Description              |
| ------------ | ---------------------------- | ------------------------ |
| **API**      | `http://localhost:8000`      | Fraud detection endpoint |
| **API Docs** | `http://localhost:8000/docs` | Swagger UI               |
| **MLflow**   | `http://localhost:5000`      | Experiment dashboard     |

The Dockerfile includes a `HEALTHCHECK` that pings `/health` every 30 seconds to ensure the API is responsive.

---

## Testing

The project includes two test suites:

### Data & Model Tests

Validates data quality (no nulls, valid ranges, reasonable fraud ratio) and model performance (accuracy ≥ 90%, F1 ≥ 0.3, non-zero precision & recall):

```bash
pytest tests/test_data_models.py -v
```

### API Integration Tests

Tests the live API endpoints for correct responses, validation rejections, and edge cases:

```bash
# Requires the API to be running on localhost:8000
pytest tests/test_api.py -v
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push to `main`/`develop` and every PR to `main`:

```
Checkout → Install deps → Generate data → Train model → Run data/model tests
    → Build Docker image → Start container → Run API tests → Cleanup
```

This ensures that every code change is validated against data quality checks, model performance thresholds, and API contract tests **before** it can be merged.

---

## Tech Stack

| Category                | Technologies                              |
| ----------------------- | ----------------------------------------- |
| **Language**            | Python 3.11                               |
| **ML Framework**        | scikit-learn (Random Forest)              |
| **API Framework**       | FastAPI + Uvicorn                         |
| **Experiment Tracking** | MLflow (Model Registry + Tracking Server) |
| **Data Validation**     | Great Expectations                        |
| **Feature Store**       | Feast (offline + online)                  |
| **Monitoring**          | Evidently AI, SciPy (KS test)             |
| **Containerization**    | Docker, Docker Compose                    |
| **CI/CD**               | GitHub Actions                            |
| **Data**                | Pandas, NumPy, Parquet                    |
| **Testing**             | pytest, httpx                             |

---

## License

This project is open-source and available for educational and portfolio purposes.
