from typing import Literal, Optional

from pydantic import AnyUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Pydantic Settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment / mode
    ENV: Literal["local", "dev", "prod"] = Field(
        default="local",
        description="Deployment environment",
    )

    # API
    FRAUD_API_KEY: str = Field(..., min_length=8)
    FRAUD_THRESHOLD: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Default fraud decision threshold (overridden by MLflow if set)",
    )

    # Database
    DATABASE_URL: str = Field(
        ...,
        description="Postgres DSN, e.g. postgres://user:pass@host:5432/db",
    )
    DB_MIN_CONN: int = 1
    DB_MAX_CONN: int = 10

    # MLflow
    MLFLOW_TRACKING_URI: AnyUrl = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI",
    )
    MLFLOW_MODEL_NAME: str = "fraud-detection-model"
    MLFLOW_MODEL_ALIAS: str = "champion"

    # Feature store / data
    REFERENCE_DATA_PATH: str = Field(
        default="data/train.csv",
        description="Path to reference data used for drift monitor",
    )

    # Rate limiting
    RATE_LIMIT_DEFAULT: str = "200/minute"
    RATE_LIMIT_PREDICT: str = "30/minute"

    # Misc
    LOG_LEVEL: str = "INFO"


# Singleton settings object
settings = Settings()