from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import holidays

app = FastAPI(title="Sales Forecasting API")

models = {}
initial_states = {}
feature_cols = ['lag1', 'lag7', 'lag30', 'rolling_mean_7', 'rolling_std_7',
                'rolling_mean_30', 'rolling_std_30', 'dayofweek', 'month', 'is_holiday']
us_holidays = holidays.US()

# Load models synchronously at startup (simpler)
try:
    for state in ['Alabama', 'Arizona', 'Arkansas', 'California']:
        models[state] = joblib.load(f"models/{state}_best.pkl")
        initial_states[state] = joblib.load(f"models/{state}_initial_state.pkl")
    print(f"Loaded models for: {list(models.keys())}")
except Exception as e:
    print("Error loading models:", e)

def xgboost_recursive_forecast(model, last_row, steps=56):
    forecasts = []
    current = last_row.copy()
    for _ in range(steps):
        X_pred = current[feature_cols].iloc[-1:].values
        pred = model.predict(X_pred)[0]
        forecasts.append(pred)
        new_row = current.iloc[-1:].copy()
        new_date = current['Date'].iloc[-1] + timedelta(days=1)
        new_row['Date'] = new_date
        new_row['Total'] = pred
        new_row['lag1'] = pred
        if len(current) >= 7:
            new_row['lag7'] = current['Total'].iloc[-7]
        else:
            new_row['lag7'] = current['Total'].iloc[0]
        if len(current) >= 30:
            new_row['lag30'] = current['Total'].iloc[-30]
        else:
            new_row['lag30'] = current['Total'].iloc[0]
        new_row['rolling_mean_7'] = (current['rolling_mean_7'].iloc[-1] * 6 + pred) / 7
        new_row['rolling_std_7'] = current['rolling_std_7'].iloc[-1]
        new_row['rolling_mean_30'] = (current['rolling_mean_30'].iloc[-1] * 29 + pred) / 30
        new_row['rolling_std_30'] = current['rolling_std_30'].iloc[-1]
        new_row['dayofweek'] = new_date.weekday()
        new_row['month'] = new_date.month
        new_row['dayofyear'] = new_date.timetuple().tm_yday
        new_row['is_holiday'] = 1 if new_date in us_holidays else 0
        current = pd.concat([current, new_row], ignore_index=True)
    return forecasts

class ForecastResponse(BaseModel):
    state: str
    forecast: List[float]
    dates: List[str]

@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(state: str, weeks: int = 8):
    if state not in models:
        raise HTTPException(404, f"State '{state}' not found. Available: {list(models.keys())}")
    model = models[state]
    last_row = initial_states[state]
    forecasts = xgboost_recursive_forecast(model, last_row, steps=weeks*7)
    start_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, weeks*7+1)]
    return ForecastResponse(state=state, forecast=forecasts, dates=dates)

@app.get("/health")
def health():
    return {"status": "ok"}