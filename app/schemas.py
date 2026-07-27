"""
Pydantic Schemas — API Request and Response Models.

Defines the expected structure and validation rules for:
1. Single customer prediction requests
2. Prediction responses
3. API health status
"""

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]


class CustomerInput(BaseModel):
    """
    Pydantic model for a single customer prediction request.
    Literal types reject invalid categorical values with a 422 before
    they can reach the model.
    """

    # Demographics
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender (Male / Female)")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether the customer is a senior citizen (1) or not (0)")
    Partner: YesNo = Field(..., description="Whether the customer has a partner (Yes / No)")
    Dependents: YesNo = Field(..., description="Whether the customer has dependents (Yes / No)")

    # Account Information
    tenure: int = Field(..., ge=0, description="Number of months the customer has stayed with the company")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="The contract term")
    PaperlessBilling: YesNo = Field(..., description="Whether the customer has paperless billing (Yes / No)")
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ] = Field(..., description="The customer's payment method")
    MonthlyCharges: float = Field(..., ge=0.0, description="The amount charged to the customer monthly")
    TotalCharges: float = Field(..., ge=0.0, description="The total amount charged to the customer")

    # Services
    PhoneService: YesNo = Field(..., description="Whether the customer has a phone service (Yes / No)")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(..., description="Whether the customer has multiple lines")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Customer's internet service provider")
    OnlineSecurity: InternetAddon = Field(..., description="Whether the customer has online security")
    OnlineBackup: InternetAddon = Field(..., description="Whether the customer has online backup")
    DeviceProtection: InternetAddon = Field(..., description="Whether the customer has device protection")
    TechSupport: InternetAddon = Field(..., description="Whether the customer has tech support")
    StreamingTV: InternetAddon = Field(..., description="Whether the customer has streaming TV")
    StreamingMovies: InternetAddon = Field(..., description="Whether the customer has streaming movies")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class PredictionResponse(BaseModel):
    """
    Pydantic model for the prediction response.
    """
    churn_prediction: int = Field(..., description="Predicted class: 1 (Churn) or 0 (No Churn)")
    churn_probability: float = Field(..., description="Probability of churning (0.0 to 1.0)")
    risk_level: str = Field(..., description="Risk category: Low, Medium, or High")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "churn_prediction": 1,
                "churn_probability": 0.78,
                "risk_level": "High"
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    """
    Pydantic model for batch prediction response.
    """
    predictions: List[PredictionResponse]
    total_processed: int


class HealthResponse(BaseModel):
    """
    Pydantic model for API health check response.
    """
    status: str = Field(..., description="API operational status")
    model_version: str = Field(..., description="Loaded model version/timestamp")
    uptime_seconds: float = Field(..., description="How long the API has been running")
    timestamp: str = Field(..., description="Current ISO-8601 timestamp")
