# BatchMind — AI Manufacturing Intelligence

Welcome to BatchMind, a causal-physics hybrid intelligence platform for pharmaceutical manufacturing optimization!

## 🚀 Key Requirements

### 1. Prerequisites
- **Python 3.10+** (tested on 3.11)
- **Node.js 18+** (tested on 18/20)
- **OpenAI API Key** (required for the Copilot feature)

---

## 🏗️ Setup Instructions

### 1. Clone the repository
Make sure you've cloned the repository to your local machine:
```bash
git clone <your-github-repo-url>
cd batchmind
```

### 2. Backend Setup
The backend is a FastAPI application powered by structural causal models (Pearl's framework), Python multi-objective optimizers (NSGA-II), and physics-based models (Heckel, Ryshkewitch, Page).

```bash
# Navigate to the backend folder
cd backend

# Create a virtual environment (optional but recommended)
python -m venv venv
# Windows: venv\\Scripts\\activate
# Mac/Linux: source venv/bin/activate

# Install the Python dependencies
pip install -r requirements.txt

# Create your .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```
*(Make sure you replace `your_openai_api_key_here` with your actual key from OpenAI. Do NOT commit this file. **`.env` is ignored by git**.)*

### 3. Frontend Setup
The frontend is a React application built with Vite, Tailwind (or custom CSS), Recharts, and Lucide Icons.

```bash
# Navigate to the frontend folder
cd ../frontend

# Install the Node dependencies
npm install
```

---

## 🏃‍♂️ Running the Application

You'll need two terminal windows open simultaneously.

### Terminal 1: Backend Server
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```
- Wait for the server to say `Application startup complete`
- You will see calibration logs for physics, SCM fitting, and DoWhy refutations during the initial startup (this might take a few seconds).
- The API will be available at: http://localhost:8000
- Auto-generated Swagger Docs cover all endpoints: http://localhost:8000/docs

### Terminal 2: Frontend Server
```bash
cd frontend
npm run dev
```
- The frontend will be available at: http://localhost:5173

---

## 🔬 Core Technologies & Modules
This is a comprehensive hackathon implementation, entirely powered by state-of-the-art causal ML engines and Bayesian frameworks:

1. **`models/scm.py`**: A true Structural Causal Model implementing Pearl's three-step Abduction-Action-Prediction counterfactuals.
2. **`models/physics.py`**: Multi-stage physics proxy engine handling Heckel Compression, Ryshkewitch strength, and Page Drying Kinetics for structural priors.
3. **`models/bayesian.py`**: Normal-Inverse-Gamma conjugate updating and credible interval trackers.
4. **`models/causal.py`**: Exposes physical equations, LightGBM surrogate constraint engines, and NSGA-II constrained optimizers.
5. **`models/fingerprint.py`**: DTW barycentric averages (DBA), coupled with Isolation Forests and Ruptures (PELT) change-point detection for anomaly scoring.

Enjoy exploring the most rigorous hackathon implementation for Causal AI + Physics out there! Let the magic happen.
