import logging
import os
from datetime import datetime, timezone

from psycopg2 import pool

logger = logging.getLogger("fraud_detection.db")

_pool: pool.ThreadedConnectionPool | None = None

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id               BIGSERIAL PRIMARY KEY,
    request_id       TEXT        NOT NULL UNIQUE,
    predicted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Input features
    amount           NUMERIC(12, 4) NOT NULL,
    hour             SMALLINT       NOT NULL,
    day_of_week      SMALLINT       NOT NULL,
    merchant_category TEXT          NOT NULL,

    -- Feast context
    feast_avg_amount    NUMERIC(12, 4),
    feast_tx_count      INTEGER,
    feast_fraud_rate    NUMERIC(8, 6),
    feast_status        TEXT NOT NULL DEFAULT 'live',

    -- Output
    is_fraud         BOOLEAN        NOT NULL,
    fraud_probability NUMERIC(8, 6)  NOT NULL,

    -- Model metadata
    model_version    TEXT,
    model_source     TEXT
);
"""

INSERT_SQL = """
INSERT INTO predictions (
    request_id, predicted_at,
    amount, hour, day_of_week, merchant_category,
    feast_avg_amount, feast_tx_count, feast_fraud_rate, feast_status,
    is_fraud, fraud_probability,
    model_version, model_source
) VALUES (
    %(request_id)s, %(predicted_at)s,
    %(amount)s, %(hour)s, %(day_of_week)s, %(merchant_category)s,
    %(feast_avg_amount)s, %(feast_tx_count)s, %(feast_fraud_rate)s, %(feast_status)s,
    %(is_fraud)s, %(fraud_probability)s,
    %(model_version)s, %(model_source)s
)
ON CONFLICT (request_id) DO NOTHING;
"""


def init_pool(min_conn: int = 1, max_conn: int = 10) -> None:
    """Create the connection pool and ensure the table exists. Call once at startup."""
    global _pool
    dsn = os.environ.get("DATABASE_URL", None)  # e.g. postgresql://user:pass@host:5432/fraud
    if not dsn:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Set it before starting the server."
        )
    _pool = pool.ThreadedConnectionPool(min_conn, max_conn, dsn)
    logger.info("DB connection pool created (min=%d max=%d)", min_conn, max_conn)
    _ensure_table()


def close_pool() -> None:
    """Close all connections in the pool. Call at shutdown."""
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None
        logger.info("DB connection pool closed")


def _ensure_table() -> None:
    conn = _pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        logger.info("predictions table ready")
    finally:
        _pool.putconn(conn)


def persist_prediction(
    *,
    request_id: str,
    amount: float,
    hour: int,
    day_of_week: int,
    merchant_category: str,
    feast_features: dict,
    feast_status: str,
    is_fraud: bool,
    fraud_probability: float,
    model_version: str,
    model_source: str,
) -> None:
    if _pool is None:
        logger.warning("DB pool not initialized — skipping persist [request_id=%s]", request_id)
        return

    conn = _pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(INSERT_SQL, {
                "request_id":        request_id,
                "predicted_at":      datetime.now(timezone.utc),
                "amount":            amount,
                "hour":              hour,
                "day_of_week":       day_of_week,
                "merchant_category": merchant_category,
                "feast_avg_amount":  feast_features.get("merchant_avg_amount"),
                "feast_tx_count":    feast_features.get("merchant_tx_count"),
                "feast_fraud_rate":  feast_features.get("merchant_fraud_rate"),
                "feast_status":      feast_status,
                "is_fraud":          is_fraud,
                "fraud_probability": fraud_probability,
                "model_version":     model_version,
                "model_source":      model_source,
            })
        logger.debug("Persisted prediction [request_id=%s]", request_id)
    except Exception:
        logger.exception("Failed to persist prediction [request_id=%s] — continuing", request_id)
    finally:
        _pool.putconn(conn)