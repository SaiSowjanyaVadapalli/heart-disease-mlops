"""Pydantic request/response schemas for the heart disease risk API."""

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(
        ..., ge=60, le=250, description="Resting blood pressure (mm Hg)"
    )
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(
        ..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true)"
    )
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (1=yes)")
    oldpeak: float = Field(
        ..., ge=0, le=10, description="ST depression induced by exercise"
    )
    slope: int = Field(
        ..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)"
    )
    ca: int = Field(
        ...,
        ge=0,
        le=4,
        description="Number of major vessels colored by fluoroscopy (0-4)",
    )
    thal: int = Field(
        ...,
        ge=0,
        le=3,
        description="Thalassemia (0=unknown, 1=normal, 2=fixed defect, 3=reversible defect)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
