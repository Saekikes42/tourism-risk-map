from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import uvicorn

app = FastAPI(title="Tourism Risk API", description="Модуль Г (4.1). API Предсказания рисков маршрутов")

MODEL_PATH = "models/risk_classifier.pkl"
try:
    model = joblib.load(MODEL_PATH)
except:
    model = None

class PredictionRequest(BaseModel):
    elevation: float
    temp: float
    speed: float
    distance: float
    water_counts: float
    vegetation_counts: float
    month: int

@app.on_event("startup")
def load_startup():
    print("=== API Поднято ===")
    if model is None:
        print("Внимание: Модель не загружена! Запустите agent_3_1_modeling.py")

@app.post("/predict_risk")
def predict_risk(data: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not trained yet.")
        
    input_df = pd.DataFrame([data.dict()])
    
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        
        return {
            "status": "success",
            "risk_cluster": int(pred),
            "confidence": float(np.max(prob)),
            "message": "Прогноз успешно построен."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("agent_4_1_api:app", host="127.0.0.1", port=8000, reload=True)
