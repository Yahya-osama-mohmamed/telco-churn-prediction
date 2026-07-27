import pytest
from fastapi.testclient import TestClient
from app.api import app, load_model

client = TestClient(app)

@pytest.fixture
def valid_payload():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 2150.0
    }

def test_health_check():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_version" in data
    assert "uptime_seconds" in data
    assert "timestamp" in data

def test_predict_endpoint_validation_error():
    """Test that missing fields return 422 Unprocessable Entity."""
    payload = {"gender": "Female"}  # Missing all other fields
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

# Note: We can't reliably test a successful /predict response here
# without a trained model file present on disk, which we don't have
# during CI before the training step runs.
def test_predict_endpoint_no_model(valid_payload, monkeypatch):
    """Test the predict endpoint when the model isn't loaded."""
    # Force load_model to return None
    monkeypatch.setattr("app.api.load_model", lambda: (None, None))
    
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 503
    assert "Model is not loaded" in response.json()["detail"]
