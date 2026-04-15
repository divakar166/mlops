import logging
import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any
from app.config import settings

from psycopg2 import pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger("fraud_detection.db")

_pool: pool.ThreadedConnectionPool | None = None

CREATE_PREDICTIONS_TABLE_SQL = """
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

CREATE INDEX IF NOT EXISTS predictions_predicted_at_idx
    ON predictions (predicted_at DESC);
"""

CREATE_DRIFT_RESULTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS drift_results (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Window metadata
    window_size     INTEGER      NOT NULL,
    current_samples INTEGER      NOT NULL,

    -- Drift summary
    drift_share     NUMERIC(8, 6) NOT NULL,
    n_features      INTEGER       NOT NULL,
    n_drifted       INTEGER       NOT NULL,
    alert           BOOLEAN       NOT NULL,

    -- Threshold used
    threshold       NUMERIC(8, 6) NOT NULL,

    -- Columns and stats (JSON blobs)
    drifted_columns JSONB         NOT NULL,
    column_stats    JSONB         NOT NULL
);

CREATE INDEX IF NOT EXISTS drift_results_created_at_idx
    ON drift_results (created_at DESC);
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


def init_pool(min_conn: int | None = None, max_conn: int | None = None) -> None:
    """Create the connection pool and ensure the table exists. Call once at startup."""
    global _pool
    dsn = settings.DATABASE_URL
    min_c = min_conn or settings.DB_MIN_CONN
    max_c = max_conn or settings.DB_MAX_CONN
    _pool = pool.ThreadedConnectionPool(min_c, max_c, dsn)
    logger.info("DB connection pool created (min=%d max=%d)", min_c, max_c)
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
            cur.execute(CREATE_PREDICTIONS_TABLE_SQL)
            cur.execute(CREATE_DRIFT_RESULTS_TABLE_SQL)
        logger.info("predictions and drift_results tables ready")
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


def get_recent_predictions(limit: int = 100) -> List[Dict[str, Any]]:
    if _pool is None:
        return []
    conn = _pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM predictions ORDER BY predicted_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception:
        logger.exception("Failed to fetch recent predictions")
        return []
    finally:
        _pool.putconn(conn)


def get_prediction_stats() -> Dict[str, Any]:
    if _pool is None:
        return {}
    conn = _pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                        AS total,
                    SUM(is_fraud::int)                             AS total_fraud,
                    ROUND(AVG(is_fraud::int) * 100, 2)             AS fraud_rate_pct,
                    ROUND(AVG(fraud_probability)  * 100, 2)        AS avg_fraud_prob_pct,
                    ROUND(AVG(amount)::numeric,   2)               AS avg_amount,
                    SUM(CASE WHEN feast_status = 'fallback'
                             THEN 1 ELSE 0 END)                    AS feast_fallbacks
                FROM predictions
            """)
            row = cur.fetchone()

            cur.execute("""
                SELECT merchant_category,
                       COUNT(*)                            AS count,
                       ROUND(AVG(is_fraud::int) * 100, 2) AS fraud_rate_pct
                FROM predictions
                GROUP BY merchant_category
                ORDER BY count DESC
            """)
            by_category = cur.fetchall()

        return {
            "total":              row["total"]              or 0,
            "total_fraud":        row["total_fraud"]        or 0,
            "fraud_rate_pct":     row["fraud_rate_pct"]     or 0.0,
            "avg_fraud_prob_pct": row["avg_fraud_prob_pct"] or 0.0,
            "avg_amount":         row["avg_amount"]         or 0.0,
            "feast_fallbacks":    row["feast_fallbacks"]    or 0,
            "by_category":        [dict(r) for r in by_category],
        }
    except Exception:
        logger.exception("Failed to fetch prediction stats")
        return {}
    finally:
        _pool.putconn(conn)

def persist_drift_result(*, result: Dict[str, Any], window_size: int) -> None:
    if _pool is None:
        logger.warning("DB pool not initialized — skipping drift persist")
        return

    conn = _pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO drift_results (
                    created_at,
                    window_size, current_samples,
                    drift_share, n_features, n_drifted, alert,
                    threshold,
                    drifted_columns, column_stats
                ) VALUES (
                    %(created_at)s,
                    %(window_size)s, %(current_samples)s,
                    %(drift_share)s, %(n_features)s, %(n_drifted)s, %(alert)s,
                    %(threshold)s,
                    %(drifted_columns)s, %(column_stats)s
                )
                """,
                {
                    "created_at":      datetime.fromisoformat(result["timestamp"]),
                    "window_size":     window_size,
                    "current_samples": result["current_samples"],
                    "drift_share":     result["drift_share"],
                    "n_features":      result["n_features"],
                    "n_drifted":       result["n_drifted"],
                    "alert":           result["alert"],
                    "threshold":       result["threshold"],
                    "drifted_columns": json.dumps(result["drifted_columns"]),
                    "column_stats":    json.dumps(result["column_stats"]),
                },
            )
        logger.debug("Persisted drift result window=%d drift_share=%.4f", window_size, result["drift_share"])
    except Exception:
        logger.exception("Failed to persist drift result — continuing")
    finally:
        _pool.putconn(conn)

def get_drift_history(limit: int = 50) -> List[Dict[str, Any]]:
    if _pool is None:
        return []

    conn = _pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    created_at,
                    window_size,
                    current_samples,
                    drift_share,
                    n_features,
                    n_drifted,
                    alert,
                    threshold,
                    drifted_columns,
                    column_stats
                FROM drift_results
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        # Convert JSONB -> Python
        history = []
        for r in rows:
            history.append(
                {
                    "timestamp":       r["created_at"].isoformat(),
                    "window_size":     r["window_size"],
                    "current_samples": r["current_samples"],
                    "drift_share":     float(r["drift_share"]),
                    "n_features":      r["n_features"],
                    "n_drifted":       r["n_drifted"],
                    "alert":           r["alert"],
                    "threshold":       float(r["threshold"]),
                    "drifted_columns": r["drifted_columns"],
                    "column_stats":    r["column_stats"],
                }
            )
        return history
    except Exception:
        logger.exception("Failed to fetch drift history")
        return []
    finally:
        _pool.putconn(conn)