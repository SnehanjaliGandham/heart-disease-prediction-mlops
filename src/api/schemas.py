"""Request and response models for the prediction API.

Every field is validated before it reaches the model: numeric values must lie in a
plausible clinical range and coded fields must use a valid UCI code. Anything else
is rejected with HTTP 422 and a per-field error message. `ca` and `thal` may be
null or omitted, because the model pipeline imputes them (as it did in training).
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Example used in the interactive /docs page and in samples/high_risk.json.
EXAMPLE_PATIENT = {
    "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0, "restecg": 2,
    "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3,
}

MAX_BATCH = 100


class PatientFeatures(BaseModel):
    """The 13 clinical attributes of the UCI Heart Disease (Cleveland) dataset."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_PATIENT]})

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true)")
    restecg: Literal[0, 1, 2] = Field(
        ..., description="Resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina (1 = yes)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(
        ..., description="Slope of peak exercise ST segment: 1 up, 2 flat, 3 down")
    ca: Optional[int] = Field(
        None, ge=0, le=3, description="Major vessels coloured by fluoroscopy (0-3); may be null")
    thal: Optional[Literal[3, 6, 7]] = Field(
        None, description="Thalassemia: 3 normal, 6 fixed defect, 7 reversible; may be null")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patients: List[PatientFeatures] = Field(..., min_length=1, max_length=MAX_BATCH)


class PredictionResult(BaseModel):
    prediction: int = Field(..., description="1 = heart disease, 0 = no heart disease")
    label: str
    probability_disease: float = Field(..., description="Model probability of heart disease")
    confidence: float = Field(..., description="Probability of the predicted class")


class PredictionResponse(PredictionResult):
    model_version: str
    request_id: str


class BatchResponse(BaseModel):
    predictions: List[PredictionResult]
    count: int
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
