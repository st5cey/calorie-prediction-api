#fastAPI uses pydantic to validate incoming JSON requests automatically (handles errors without crashing model)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

#initialize app
cal_app = FastAPI(
    title="Calories Burned Prediction API",
    description="Production endpoint for predicting workout caloric burn using XGBoost.",
    version="1.0.0"
)

#define expected JSON input schema using Pydantic
class WorkoutPayload(BaseModel):
    Gender: str = Field(..., example="male")
    Age: int = Field(..., ge=1, le=120, example=28)
    Height: float = Field(..., ge=50.0, le=250.0, example=178.0) # in cm
    Weight: float = Field(..., ge=20.0, le=300.0, example=75.0)  # in kg
    Duration: float = Field(..., ge=1.0, le=300.0, example=25.0) # in minutes
    Heart_Rate: float = Field(..., ge=40.0, le=220.0, example=130.0)
    Body_Temp: float = Field(..., ge=35.0, le=43.0, example=40.0) # in Celsius

#load pipeline
try:
    pipeline = joblib.load('xgboost_calories_pipeline.pkl')
except Exception as e:
    pipeline = None

#feature engineering helper
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    df_feat = data.copy()
    df_feat['BMI'] = df_feat['Weight'] / ((df_feat['Height'] / 100) ** 2)
    df_feat['Workload_Index'] = df_feat['Heart_Rate'] * df_feat['Duration']
    df_feat['Temp_Elevation'] = df_feat['Body_Temp'] - 37.0
    return df_feat

#define health check endpoint
@cal_app.get("/")
def home():
    return {"status": "online", "message": "Calorie Prediction API is running."}

#define prediction endpoint
@cal_app.post("/predict")
def predict_calories(payload: WorkoutPayload):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline not found on server.")

    #convert payload dict to DataFrame
    raw_data = pd.DataFrame([payload.model_dump()])

    #feature engineering & prediction
    engineered_data = engineer_features(raw_data)
    prediction = pipeline.predict(engineered_data)[0]
    
    return {
        "predicted_calories_burnt": round(float(prediction), 2),
        "status": "success"
    }