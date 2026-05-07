# Sales Forecasting System – 8‑Week Forecast by State

## 📌 Project Overview

This project builds a **production‑ready forecasting system** that predicts the next 8 weeks of sales for each US state using historical daily sales data.  
It implements and compares **four different models** (SARIMA, Prophet, XGBoost, LSTM), automatically selects the best performing model per state, and exposes the forecasts via a **REST API**.

---

## 🎯 Features

- Handles **missing dates** and **missing values** by linear interpolation.
- Extracts **seasonality** and **trend** using lag features, rolling statistics, date components, and holiday flags.
- Trains and evaluates **ARIMA/SARIMA, Facebook Prophet, XGBoost (with lag features), and LSTM**.
- Uses **time series cross‑validation** (last 56 days as validation set) – no leakage.
- Automatically selects the best model for each state based on **lowest validation RMSE**.
- Saves the trained models and exposes **8‑week forecasts** via a **FastAPI** endpoint.
- Includes a full **Jupyter notebook** that demonstrates all steps.

---

## 📂 Repository Structure

```
.
├── data/                                    # original dataset (not included in repo)
├── notebooks/
│   └── forecasting_pipeline.ipynb       # complete training & selection pipeline
├── models/                                  # saved best models + initial states
│   ├── Alabama_best.pkl
│   ├── Alabama_initial_state.pkl
│   ├── Arizona_best.pkl
│   ├── ...
├── api.py                                   # FastAPI server
├── requirements.txt                         # dependencies
├── README.md                                # this file

```

---

## 🚀 Setup & Installation

### 1. Clone / Download the project

Place the provided files in a local folder.

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> `requirements.txt` includes: `pandas numpy matplotlib seaborn statsmodels prophet xgboost tensorflow scikit-learn pmdarima holidays fastapi uvicorn joblib`

### 4. Prepare the data

Place the Excel file `Forecasting Case-Study.xlsx` inside the `data/` folder.  
The notebook and API expect this path.

---

## 📓 Training & Model Selection (Jupyter Notebook)

Open `notebooks/forecasting_pipeline.ipynb` and run all cells.  
The notebook will:

- Load and clean the data.
- Fill missing dates using `asfreq('D')` and linear interpolation.
- Engineer features:
  - Lags (t-1, t-7, t-30)
  - Rolling mean & std (7‑day, 30‑day)
  - Day of week, month, day of year
  - US holiday flag
- Split each state's data into **train** (all but last 56 days) and **validation** (last 56 days).
- Train **SARIMA**, **Prophet**, **XGBoost** (with lag features), and **LSTM** on the train set.
- Evaluate each model on the validation set (RMSE).
- Select the model with the lowest RMSE per state.
- Retrain the selected model on **all historical data** (train + validation).
- Save the retrained model and the last feature row (initial state) to the `models/` folder.

After running the notebook, the `models/` folder will contain files like:

- `California_best.pkl` – the best model object
- `California_initial_state.pkl` – the last known feature row (used for recursive forecasting)

> **Note:** Training all 50+ states can take several hours. For demonstration, the notebook can be limited to a few states (see `demo_states` list inside). The pre‑trained models for the first 4 states are already provided.

---

## 🌐 REST API – Serving Forecasts

### Start the API server

From the root project folder (where `api.py` is located), run:

```bash
uvicorn api:app --reload
```

You will see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Endpoint

```
GET /forecast?state=<state_name>&weeks=<number_of_weeks>
```

- `state` – name of the US state (case‑sensitive, e.g., `California`, `Alabama`).
- `weeks` – number of weeks to forecast (default is 8). The API returns `weeks * 7` daily forecasts.

### Example request (using `curl`)

```bash
curl "http://localhost:8000/forecast?state=California&weeks=8"
```

### Example response (JSON)

```json
{
  "state": "California",
  "forecast": [215825008.0, 214994240.0, ...],
  "dates": ["2026-05-08", "2026-05-09", ...]
}
```

### Testing in browser

Open `http://localhost:8000/forecast?state=Alabama&weeks=8` – you will see the same JSON response.

### Health check

`GET /health` returns `{"status": "ok"}`.

---

## 📊 Results Summary (for first 4 states)

| State      | Best Model | Validation RMSE |
|------------|------------|----------------|
| Alabama    | XGBoost    | 3,186,720      |
| Arizona    | XGBoost    | 3,356,954      |
| Arkansas   | XGBoost    | 1,899,886      |
| California | XGBoost    | 10,314,252     |

> XGBoost with lag features outperformed SARIMA, Prophet, and LSTM on these states. The forecasts for the next 8 weeks are stable and follow the historical trend.

---


## 🛠️ Technology Stack

- **Data processing**: Pandas, NumPy, Holidays
- **Models**: `pmdarima` (SARIMA), `prophet`, `xgboost`, `tensorflow` (LSTM)
- **Evaluation**: scikit‑learn (RMSE)
- **API**: FastAPI, Uvicorn
- **Serialisation**: joblib, pickle

---
