# LoanLens — Loan Default Prediction & Risk Intelligence Platform

> **Academic Project Notice:** LoanLens is an educational demonstration of credit risk modeling on Kaggle data. Risk scores and model classifications are statistical outputs for educational, evaluative, and simulation purposes only — they do not constitute actual commercial lending decisions or professional financial advice.

[![Node.js](https://img.shields.io/badge/Node.js-18%2B-339933?style=flat-square&logo=node.js&logoColor=white)](https://nodejs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![Flask](https://img.shields.io/badge/Flask-2.2%2B-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com)
[![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=flat-square&logo=render&logoColor=white)](https://render.com)

**LoanLens** is an end-to-end Machine Learning web application that evaluates loan default risk across 255,347 historical consumer credit records. The platform couples a production-calibrated **L2-Regularized Logistic Regression** model with interactive underwriting simulation, multi-model benchmarking, and exploratory data analysis — deployed as a unified Docker container on Render.

**Live Demo:** [https://loanlens-hynm.onrender.com](https://loanlens-hynm.onrender.com)

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Features](#2-features)
3. [Machine Learning Models](#3-machine-learning-models)
4. [Dataset & Preprocessing](#4-dataset--preprocessing)
5. [Risk Score & Transformation Methodology](#5-risk-score--transformation-methodology)
6. [API Endpoint Documentation](#6-api-endpoint-documentation)
7. [Project Structure](#7-project-structure)
8. [Environment Variables](#8-environment-variables)
9. [Local Development Setup](#9-local-development-setup)
10. [Production Deployment](#10-production-deployment)

---

## 1. Architecture

LoanLens uses a decoupled two-tier architecture inside a single Docker container:

- **Frontend Layer**: Express.js (Node.js/TypeScript) serving server-rendered EJS views with Vanilla CSS and Chart.js visualization
- **ML Inference Engine**: Python Flask microservice loading pre-trained scikit-learn pipelines at startup
- **Inter-Service Communication**: Node.js proxy routes (/api/*) forwarding to the Flask backend over localhost

`
Browser Client
      |
      | HTTP :3000
      v
Express.js (TypeScript)   ─── EJS Pages & Static Assets
      |
      | /api/predict, /api/models, /api/data/insights
      v
Flask ML Engine           ─── HTTP :5001
      |
      v
  Scikit-Learn Artifacts
  ├── preprocessor.pkl
  ├── logistic_regression.pkl   (production)
  ├── knn.pkl
  ├── naive_bayes.pkl
  └── decision_tree.pkl
`

---

## 2. Features

| Feature | Description |
|---|---|
| **Risk Prediction** | 16-feature loan application scored through L2 Logistic Regression with calibrated default probability |
| **Risk Score** | Institutional 0–100 creditworthiness score inverted from default probability |
| **Factor Analysis** | Per-prediction log-odds coefficient breakdown identifying protective and risk-amplifying features |
| **Model Benchmarking** | 4-model comparison table with ROC-AUC, 5-fold CV AUC, precision, recall, and confusion matrices |
| **Scenario Simulator** | Interactive borrower scenario editor for what-if analysis |
| **EDA Dashboard** | Dataset distribution charts, correlation matrix, and dataset summary |
| **Prediction History** | Session-scoped prediction log with re-inspection |

---

## 3. Machine Learning Models

LoanLens evaluates four primary classification algorithms on the 255k dataset, plus one from-scratch reference implementation:

| Model ID | Algorithm | Family | ROC-AUC | 5-Fold CV AUC | Default Recall | Status |
|---|---|---|---|---|---|---|
| **logistic-regression** | L2-Regularized Logistic Regression (Balanced) | LogisticRegression | **0.7532** | **0.7426 ± 0.006** | **69.95%** | **✅ Production Model** |
| **knn** | Distance-Weighted K-Nearest Neighbors (k=25) | KNeighborsClassifier | 0.6927 | 0.6753 ± 0.007 | 0.70% | Benchmark |
| **naive-bayes** | Gaussian Naive Bayes (ar_smoothing=1e-5) | GaussianNB | 0.7499 | 0.7396 ± 0.004 | 2.56% | Benchmark |
| **decision-tree** | Entropy-Partitioned Decision Tree (max_depth=6) | DecisionTreeClassifier | 0.7254 | 0.7036 ± 0.006 | 3.02% | Benchmark |
| *scratch-logistic* | Batch Gradient Descent (Pure NumPy) | NumPy | 0.7530 | 0.7530 ± 0.003 | 68.40% | Reference |

### Why Logistic Regression?

In retail lending, false negatives (approving an applicant who defaults) are vastly more costly than false positives. The balanced Logistic Regression model achieves:

- **69.95% Default Recall** — correctly flagging 4,149 of 5,931 test defaults
- **Highest ROC-AUC (0.7532)** — best overall discriminative ranking power
- **Interpretable Coefficients** — log-odds transparency for audit and explainability

---

## 4. Dataset & Preprocessing

| Property | Value |
|---|---|
| **Source** | Kaggle — 
ikhil1e9/loan-default |
| **Total Records** | 255,347 clean records |
| **Duplicates / Missing** | 0 / 0 |
| **Target: Non-Default (0)** | 225,694 (88.39%) |
| **Target: Default (1)** | 29,653 (11.61%) |
| **Raw Features** | 16 (9 continuous, 7 categorical) |
| **Encoded Dimension** | 24 features post-preprocessing |

### Preprocessing Pipeline (rtifacts/preprocessing/preprocessor.pkl)

**Continuous Features (9)** — normalized via StandardScaler(with_mean=True, with_std=True):
Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio

**Categorical Features (7)** — encoded via OneHotEncoder(drop='first', sparse_output=False):
Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner

---

## 5. Risk Score & Transformation Methodology

### Step 1 — Raw Model Output
The Logistic Regression produces a calibrated default probability:

`
P(Default = 1 | X) = sigmoid(w^T * X + b)  ∈ [0.0000, 1.0000]
`

### Step 2 — Institutional Risk Score
The probability is inverted and scaled to a 0–100 creditworthiness score:

`
risk_score = round((1.0 - P(Default)) × 100)  ∈ [0, 100]
`

*Higher scores indicate lower default risk and greater creditworthiness.*

### Step 3 — Threshold Classification

| Risk Level | Probability Range | Risk Score Range | Classification |
|---|---|---|---|
| **Low Risk** | P < 0.28 | 73 – 100 | No Default (0) |
| **Moderate Risk** | 0.28 ≤ P < 0.50 | 51 – 72 | No Default (0) |
| **High Risk** | P ≥ 0.50 | 0 – 50 | Default (1) |

---

## 6. API Endpoint Documentation

All API routes are proxied through Express.js to the Flask ML backend.

| Method | Path | Description |
|---|---|---|
| GET | /api/health | Service health check |
| POST | /api/predict | Run full 16-feature prediction |
| GET | /api/models | List all 4 models with summary metrics |
| GET | /api/models/comparison | 4-model evaluation table + ROC curve data |
| GET | /api/models/:modelId | Parameters, confusion matrix, and coefficients for a specific model |
| GET | /api/data/insights | EDA distributions, correlation matrix, dataset summary |

### POST /api/predict — Example

**Request:**
`json
{
  "Age": 42,
  "Income": 115000,
  "LoanAmount": 25000,
  "CreditScore": 780,
  "MonthsEmployed": 64,
  "NumCreditLines": 3,
  "InterestRate": 6.8,
  "LoanTerm": 36,
  "DTIRatio": 0.22,
  "Education": "Master's",
  "EmploymentType": "Full-time",
  "MaritalStatus": "Married",
  "HasMortgage": "Yes",
  "HasDependents": "No",
  "LoanPurpose": "Home",
  "HasCoSigner": "No"
}
`

**Response (200 OK):**
`json
{
  "prediction": 0,
  "label": "No Default",
  "probability": 0.1148,
  "confidence": 0.8852,
  "risk_score": 89,
  "risk_level": "Low",
  "model_used": "Logistic Regression",
  "factors": [
    {
      "feature": "InterestRate",
      "label": "Contract Interest Rate",
      "value": 6.8,
      "benchmark": "10.0%",
      "direction": "down",
      "impact": 25,
      "raw_impact": -0.5023,
      "type": "mitigant",
      "desc": "Contract Interest Rate: -0.5 log-odds (Protective Factor)"
    }
  ],
  "recommendation": {
    "action": "Predicted Default Risk: Low (Model Classification: No Default)",
    "points": [
      "Model predicts strong repayment indicators with estimated default probability below 0.28.",
      "Solid financial fundamentals (healthy DTI, prime credit rating, and stable employment tenure) serve as strong protective factors.",
      "Notice: This prediction is an ML model output for demonstration purposes and does not constitute financial advice."
    ]
  },
  "success": true
}
`

---

## 7. Project Structure

`
LoanLens App/
├── artifacts/
│   ├── metadata/
│   │   ├── model_context.json        # Training config & hyperparameters
│   │   └── selected_model.json       # Production model selection record
│   ├── metrics/
│   │   └── all_metrics.json          # Full 4-model evaluation metrics
│   ├── models/
│   │   ├── logistic_regression.pkl   # Production model
│   │   ├── knn.pkl
│   │   ├── naive_bayes.pkl
│   │   └── decision_tree.pkl
│   └── preprocessing/
│       └── preprocessor.pkl          # Fitted ColumnTransformer pipeline
├── flask_backend/
│   └── app.py                        # Flask ML inference API
├── public/
│   ├── css/                          # Vanilla CSS stylesheets
│   └── js/                           # Client-side JavaScript (charts, predict, simulator)
├── scripts/
│   └── build_artifacts.py            # Script to retrain and serialize .pkl artifacts
├── src/
│   ├── services/                     # TypeScript service layer (prediction, model, analytics)
│   └── types/                        # TypeScript type definitions
├── views/
│   ├── components/                   # Reusable EJS partials
│   └── *.ejs                         # Page templates (index, predict, models, simulator, history)
├── Dockerfile                        # Unified Node + Flask container
├── render.yaml                       # Render deployment config
├── requirements.txt                  # Python dependencies
├── package.json                      # Node.js dependencies & scripts
├── server.ts                         # Express.js entry point
└── Loan_Default_Prediction.ipynb     # Full ML training notebook
`

---

## 8. Environment Variables

| Variable | Description | Development Default | Production |
|---|---|---|---|
| PORT | Web server port | 3000 | Set by cloud provider |
| HOST | Bind address |  .0.0.0 |  .0.0.0 |
| USE_REMOTE_BACKEND | Enable Flask proxy | 	rue | 	rue |
| FLASK_BACKEND_URL | Flask service URL | http://127.0.0.1:5001 | http://127.0.0.1:5001 |
| FLASK_PORT | Flask listen port | 5001 | 5001 |
| FLASK_HOST | Flask bind address | 127.0.0.1 | 127.0.0.1 |
| ALLOWED_ORIGINS | CORS allowed origins | * | Deployed frontend domain |

---

## 9. Local Development Setup

### Prerequisites

- Node.js v18+ and npm
- Python 3.10+ with pip

### Step 1 — Clone & Install

`ash
git clone https://github.com/Bhaviik-5202/LoanLens.git
cd LoanLens

# Install Node dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt
`

### Step 2 — Start Development Server

`ash
npm run dev
`

The Express server starts on **http://localhost:3000** and automatically launches the Flask backend on port 5001 as a managed child process.

### Step 3 — (Optional) Rebuild ML Artifacts

If you need to retrain and re-serialize the .pkl artifacts from the dataset:

`ash
python scripts/build_artifacts.py
`

---

## 10. Production Deployment

### Option A — Docker (Recommended)

The app is pre-configured as a single unified Docker container with both Node.js and Flask running together.

`ash
docker build -t loanlens .
docker run -p 3000:3000 \
  -e NODE_ENV=production \
  -e PORT=3000 \
  -e HOST=0.0.0.0 \
  -e USE_REMOTE_BACKEND=true \
  -e FLASK_BACKEND_URL=http://127.0.0.1:5001 \
  loanlens
`

### Option B — Render (Current Production)

The repository includes a ender.yaml for automatic Render deployment using the Dockerfile.

1. Push to the main branch of [https://github.com/Bhaviik-5202/LoanLens](https://github.com/Bhaviik-5202/LoanLens)
2. Render detects changes and rebuilds the Docker image automatically
3. The health check at /api/health confirms successful startup

**Live URL:** [https://loanlens-hynm.onrender.com](https://loanlens-hynm.onrender.com)

### Option C — Decoupled Multi-Service

Deploy Node.js and Flask as separate services:

**Flask Backend:**
`ash
HOST=0.0.0.0 PORT=5001 python flask_backend/app.py
`

**Express Frontend:**
`ash
npm run build
USE_REMOTE_BACKEND=true FLASK_BACKEND_URL=https://<your-flask-url> npm start
`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Express.js 4 + TypeScript 5.7 |
| Templating | EJS 3 (Server-Side Rendering) |
| Styling | Vanilla CSS |
| Charts | Chart.js |
| ML Framework | scikit-learn 1.6.1 |
| ML API | Python Flask 2.2+ |
| Serialization | joblib |
| Containerization | Docker |
| Deployment | Render (Docker runtime) |
| Dataset | Kaggle — nikhil1e9/loan-default |

---

*Built as an end-to-end ML engineering project for academic demonstration of production credit risk modeling.*
