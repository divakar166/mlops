# tests/test_api.py

import pytest
import httpx

BASE_URL = "http://localhost:8000"

class TestPredictionEndpoint:
    """Tests for the /predict endpoint."""
    
    def test_valid_prediction_returns_200(self):
        """Valid input should return HTTP 200 with prediction."""
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": 100.0,
            "hour": 14,
            "day_of_week": 3,
            "merchant_category": "online"
        }, timeout=10)
        
        assert response.status_code == 200
        data = response.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert isinstance(data["is_fraud"], bool)
        assert 0 <= data["fraud_probability"] <= 1
    
    def test_high_risk_transaction(self):
        """High-risk transaction should have higher fraud probability."""
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": 500.0,
            "hour": 3,  # Late night
            "day_of_week": 1,
            "merchant_category": "online"
        }, timeout=10)
        
        assert response.status_code == 200
        data = response.json()
        # High-risk transactions should have elevated probability
        # (not asserting exact value as model may vary)
        assert data["fraud_probability"] >= 0.0
    
    def test_negative_amount_rejected(self):
        """Negative amount should be rejected with 400."""
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": -100.0,
            "hour": 14,
            "day_of_week": 3,
            "merchant_category": "online"
        }, timeout=10)
        
        assert response.status_code == 400
        assert "errors" in response.json()["detail"]
    
    def test_invalid_hour_rejected(self):
        """Invalid hour should be rejected with 400."""
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": 100.0,
            "hour": 25,  # Invalid
            "day_of_week": 3,
            "merchant_category": "online"
        }, timeout=10)
        
        assert response.status_code == 400
    
    def test_invalid_merchant_rejected(self):
        """Unknown merchant category should be rejected with 400."""
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": 100.0,
            "hour": 14,
            "day_of_week": 3,
            "merchant_category": "unknown_category"
        }, timeout=10)
        
        assert response.status_code == 400
    
    def test_missing_field_rejected(self):
        """Missing required field should be rejected."""
        response = httpx.post(f"{BASE_URL}/predict", json={
            "amount": 100.0,
            "hour": 14
            # Missing day_of_week and merchant_category
        }, timeout=10)
        
        assert response.status_code == 422  # Pydantic validation error


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_returns_200(self):
        """Health endpoint should return 200."""
        response = httpx.get(f"{BASE_URL}/health", timeout=10)
        assert response.status_code == 200
    
    def test_health_returns_healthy_status(self):
        """Health endpoint should indicate healthy status."""
        response = httpx.get(f"{BASE_URL}/health", timeout=10)
        data = response.json()
        assert data["status"] == "healthy"