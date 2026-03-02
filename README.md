# ⚡ BatchMind — AI-Driven Manufacturing Intelligence

> **Causal-Physics Hybrid Intelligence for Zero-Compromise Manufacturing**  
> 

---

## 🎯 What is BatchMind?

BatchMind is a next-generation manufacturing intelligence platform that goes beyond correlation to deliver **causal explanations** and **counterfactual interventions** for pharmaceutical tablet manufacturing.

**The core insight:** Most AI tells you *what* will happen. BatchMind tells you *why* — and *what to change* to get the outcome you want.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **⚡ Phase-Aware Energy Fingerprinting** | Learns the normal power "DNA" for each production phase. Deviations = equipment anomalies. No fault labels needed. |
| **⚗️ Causal Multi-Objective Optimizer** | Uses causal DAG + do-calculus to recommend parameter interventions — not just correlations. |
| **🎯 Bayesian Golden Signature** | Maintains optimal parameter distributions with 95% confidence intervals. Self-updates when new batches beat current best. |
| **🌱 Carbon Budget Ledger** | Per-batch CO₂e accounting (India grid: 0.82 kg/kWh), monthly budget tracking, and savings forecasting. |
| **🤖 AI Explanations** | GPT-4o powered natural language explanations of counterfactual recommendations. |

---

## 🏗️ Architecture

```
batchmind/
├── backend/          # FastAPI + Python ML
│   ├── main.py
│   ├── data_loader.py
│   ├── models/
│   │   ├── fingerprint.py     # Phase energy fingerprinting
│   │   ├── causal.py          # Causal optimizer + counterfactuals
│   │   ├── golden_signature.py # Bayesian golden signature
│   │   └── carbon.py          # Carbon ledger
│   └── routers/
│       ├── batches.py
│       ├── fingerprint.py
│       ├── optimizer.py
│       ├── golden.py
│       └── carbon.py
└── frontend/         # React (Vite)
    └── src/
        ├── pages/    # 5 pages: Dashboard, Batch, Optimizer, Golden, Carbon
        └── api.js    # API client
```

---

## ⚙️ Setup & Run

### Backend

```bash
cd backend
pip install -r requirements.txt

# Add your OpenAI API key to .env
echo "OPENAI_API_KEY=your_key_here" > .env

# Start server
python -m uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open: **http://localhost:5173**  
API Docs: **http://localhost:8000/docs**

---

## 📊 Data

- `_h_batch_process_data.xlsx` — 60 batch sheets, minute-level sensor readings (Power, Temp, Pressure, Vibration)
- `_h_batch_production_data.xlsx` — Quality outcomes (Hardness, Dissolution Rate, Friability, etc.)

---

## 🔬 Technical Innovations

### 1. Phase-Aware Energy Fingerprinting
- Normalizes power consumption time-series per phase to 100 time steps
- Computes mean ± σ band (the "fingerprint") across all 60 batches
- Z-score anomaly detection: deviation from fingerprint → Asset Health Index (0-100)

### 2. Causal Multi-Objective Optimization
- Causal DAG encodes manufacturing domain knowledge
- Linear causal models trained per target variable
- Counterfactual: finds minimum-energy batch achieving equivalent quality
- GPT-4o generates operator-friendly explanations

### 3. Bayesian Golden Signature
- Scores each batch against user-defined objective weights
- Top-10 similar batches → bootstrap confidence intervals
- Self-update mechanism: compares new batch score vs reference batch
- Pre-computed signatures: Best Quality, Best Energy, Balanced, Sustainability

### 4. Carbon Budget Ledger
- India electricity emission factor: 0.82 kg CO₂e/kWh
- Per-batch: ∫ Power_kW × dt → kWh → kg CO₂e
- Monthly budget allocation + linear trend forecasting

---

## 🎨 UI

- Dark & Light themes (persistent across sessions)
- Mobile responsive (sidebar collapses on mobile)
- Built with React + Recharts + Lucide icons
- Inter font via Google Fonts

---

## 🏆 Hackathon

**Event:** IIT Hyderabad AI Manufacturing Hackathon  
**Track:** Both Tracks — Predictive Modelling + Optimization Engine  
**Team:** [Your Name]  
**Deadline:** March 2, 2026
