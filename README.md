<div align="center">

# ⚡ PowerGrid-ExplainableAI
### Explainable Artificial Intelligence for Rural Microgrid & Battery Fleet Management

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-6.x-646CFF?logo=vite&logoColor=white)](https://vitejs.dev)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CNN%2BLSTM-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Dispatch_Classifier-EB5424)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A full-stack microgrid management platform combining probabilistic CNN+LSTM renewable forecasting, NASA battery fleet degradation diagnostics, and transparent XGBoost + SHAP explainable dispatch intelligence.

</div>

---

## 1. What This Project Does

XplainableAI helps monitor and optimize microgrid performance by combining three capabilities:

- Probabilistic energy forecasting for solar, wind, and load demand.
- Battery fleet health analysis with anomaly-aware metrics.
- Explainable dispatch recommendations with driver-level reasoning.

The frontend is built for decision visibility, while the backend orchestrates data, model outputs, and API responses.

---

## 2. System Snapshot

```mermaid
flowchart LR
  A[Frontend - React + Vite] -->|/api| B[Backend - Flask]
  B --> C[CNN+LSTM Forecast Outputs]
  B --> D[Battery Degradation Outputs]
  B --> E[Dispatch Engine Outputs]
  E --> F[SHAP Explanations]
  B --> G[Unified JSON Endpoints]
  G --> A
```

---

## 3. Monorepo Layout

```text
PowerGrid-ExplainableAI/
  run.py                                 Unified 1-command full-stack launcher
  start.bat                              Windows 1-click startup script
  backend/                               Flask API server & PostgreSQL ORM
  frontend/                              React + Vite glassmorphic dashboard app
  CNN+LSTM-FORECASTER/                   48h solar/wind/load CNN+LSTM models & predictions
  dispatch_engine/                       XGBoost dispatch classifier & SHAP explanations
  ZerodayExplorers_BatteryDegradation/   NASA battery degradation models & diagnostic charts
```

---

## 4. Frontend Overview

### Main Purpose

The frontend delivers a production-style dashboard for operators and analysts to:

- View forecast trends and uncertainty bands.
- Monitor battery health and anomaly indicators.
- Inspect dispatch mode confidence and recommendations.
- Browse explainability artifacts from model outputs.

### Frontend Stack

- React
- Vite
- Tailwind CSS
- React Router
- Recharts
- REST API Client (PostgreSQL Auth)

### Frontend Routes

- /landing
- /
- /forecast
- /battery
- /dispatch
- /explainability
- /scenarios
- /settings

---

## 5. Backend Overview

### Main Purpose

The backend aggregates model outputs, manages PostgreSQL database persistence (users, configurations, battery fleet, forecasts, dispatch history), and exposes API endpoints consumed by the frontend.

### Backend Stack

- Flask
- Flask-CORS
- PostgreSQL 18 (`psycopg2-binary`)
- PyJWT
- Pandas
- NumPy
- XGBoost
- SHAP
- scikit-learn
- Gunicorn

### API Base

- http://localhost:5000/api

### Key Endpoints

- GET /health
- GET /dashboard
- GET /forecast
- GET /forecast/summary
- GET /battery
- GET /battery/fleet
- GET /battery/cycles
- GET /battery/ids
- GET /battery/plots/<filename>
- POST /dispatch/predict
- GET /dispatch/explanations
- GET /explain/images/<filename>
- GET /explain/available

---

## 6. Quickstart & Local Development

### Option A: ⚡ Run Entire Full-Stack in 1 Command

You can launch both the **Flask Backend (Port 5000)** and **React Frontend (Port 5173)** simultaneously from the repository root:

```powershell
python run.py
```


### Option B: Run Services Manually (Separate Terminals)

#### Terminal 1: Backend
```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
*Backend URL: `http://localhost:5000`*

#### Terminal 2: Frontend
```powershell
cd frontend
npm install
npm run dev
```
*Frontend URL: `http://localhost:5173`*

---

## 7. Environment & Database Configuration

### PostgreSQL Connection
The backend automatically connects to local PostgreSQL if available, or falls back to local datasets and offline demo mode:

```env
PGHOST=localhost
PGPORT=5432
PGUSER=postgres
PGPASSWORD=1234
PGDATABASE=xplainable_ai
```

On startup, `database.py` automatically initializes tables (`users`, `user_settings`, `battery_fleet`, `forecast_records`, `dispatch_history`) and seeds the data if the tables are empty.

### Evaluator Demo Credentials
| Field | Value |
| :--- | :--- |
| **Email** | `demo@xplainable.ai` |
| **Password** | `demo123` |
| **1-Click Quick Demo** | Click the **⚡ Quick Demo Access (1-Click)** button on the landing page for immediate evaluator exploration. |

---

## 8. Data and Model Dependencies

The backend reads artifacts from sibling folders in this workspace.

Required files and folders include:

- CNN+LSTM-FORECASTER/data/bihar_predictions.csv
- dispatch_engine/outputs/dispatch_engine.ubj
- dispatch_engine/outputs/per_sample_explanations.csv
- ZerodayExplorers_BatteryDegradation/outputs/data/battery_full_summary.csv
- ZerodayExplorers_BatteryDegradation/outputs/data/nasa_2yr_all_batteries.csv
- ZerodayExplorers_BatteryDegradation/outputs/plots

If these are missing, related endpoints return empty objects/lists or fallback errors.

---

## 9. Build Commands

### Frontend Production Build

```powershell
cd frontend
npm run build
npm run preview
```

### Backend Production Start (example)

```powershell
cd backend
gunicorn "app:create_app()"
```

---

## 10. Project Notes

- Backend and frontend are decoupled and can be deployed independently.
- API prefix is consistently /api for all backend modules.
- Explainability endpoints can serve both image and html artifacts.

---

## 11. Contribution Workflow

1. Create a feature branch.
2. Keep backend and frontend changes scoped and documented.
3. Validate frontend build and backend startup before opening a pull request.
4. Include endpoint or UI impact notes in your commit/PR message.

---

## 12. License

MIT License

Copyright (c) 2026 XplainableAI Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
