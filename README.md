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
11. [Tech Stack](#11-tech-stack)

---

## 1. Architecture

LoanLens uses a decoupled two-tier architecture inside a single Docker container:

- **Frontend Layer**: Express.js (Node.js / TypeScript) serving server-rendered EJS views with Vanilla CSS and Chart.js visualizations
- **ML Inference Engine**: Python Flask microservice loading pre-trained scikit-learn pipelines once at startup
- **Inter-Service Communication**: Node.js proxy routes (`/api/*`) forwarding requests to Flask over localhost

```
Browser Client
      |
      | HTTP :3000
      v
Express.js (TypeScript)       EJS Pages & Static Assets
      |
      | /api/predict
      | /api/models
      | /api/data/insights
      v
Flask ML Engine               HTTP :5001
      |
      v
  Scikit-Learn Artifacts
      artifacts/preprocessing/preprocessor.pkl
      artifacts/models/logistic_regression.pkl   <-- Production
      artifacts/models/knn.pkl
      artifacts/models/naive_bayes.pkl
      artifacts/models/decision_tree.pkl
```

---

## 2. Features

| Feature | Description |
|---|---|
| **Risk Prediction** | 16-feature loan application scored through L2 Logistic Regression with calibrated default probability |
| **Risk Score (0–100)** | Institutional creditworthiness score inverted from default probability |
| **Factor Analysis** | Per-prediction log-odds coefficient breakdown identifying protective and risk-amplifying features |
| **Model Benchmarking** | 4-model comparison with ROC-AUC, 5-fold CV AUC, precision, recall, and confusion matrices |
| **Scenario Simulator** | Interactive borrower scenario editor for what-if analysis |
| **EDA Dashboard** | Dataset distribution charts, correlation matrix, and dataset summary statistics |
| **Prediction History** | Session-scoped prediction log with re-inspection capability |

---

## 3. Machine Learning Models

LoanLens evaluates four primary classification algorithms on the 255k dataset, plus one from-scratch reference implementation:

| Model ID | Algorithm | Family | ROC-AUC | 5-Fold CV AUC | Default Recall | Status |
|---|---|---|---|---|---|---|
| **logistic-regression** | L2-Regularized Logistic Regression (Balanced) | `LogisticRegression` | **0.7532** | **0.7426 ± 0.006** | **69.95%** | **✅ Production Model** |
| **knn** | Distance-Weighted K-Nearest Neighbors (k=25) | `KNeighborsClassifier` | 0.6927 | 0.6753 ± 0.007 | 0.70% | Benchmark |
| **naive-bayes** | Gaussian Naive Bayes (`var_smoothing=1e-5`) | `GaussianNB` | 0.7499 | 0.7396 ± 0.004 | 2.56% | Benchmark |
| **decision-tree** | Entropy-Partitioned Decision Tree (`max_depth=6`) | `DecisionTreeClassifier` | 0.7254 | 0.7036 ± 0.006 | 3.02% | Benchmark |
| *scratch-logistic* | Batch Gradient Descent (Pure NumPy) | `NumPy` | 0.7530 | 0.7530 ± 0.003 | 68.40% | Reference |

### Why Logistic Regression?

In retail lending, false negatives (approving an applicant who defaults) are vastly more costly than false positives. The balanced Logistic Regression model achieves:

- **69.95% Default Recall** — correctly flagging 4,149 of 5,931 test defaults
- **Highest ROC-AUC (0.7532)** — best overall discriminative ranking power across all 4 models
- **Interpretable Coefficients** — log-odds transparency for audit and regulatory explainability

---

## 4. Dataset & Preprocessing

| Property | Value |
|---|---|
| **Source** | Kaggle — `nikhil1e9/loan-default` |
| **Total Records** | 255,347 clean records |
| **Duplicates / Missing** | 0 / 0 |
| **Target: Non-Default (0)** | 225,694 (88.39%) |
| **Target: Default (1)** | 29,653 (11.61%) |
| **Raw Features** | 16 (9 continuous, 7 categorical) |
| **Encoded Dimension** | 24 features post-preprocessing |

### Preprocessing Pipeline (`artifacts/preprocessing/preprocessor.pkl`)

**Continuous Features (9)** — normalized via `StandardScaler(with_mean=True, with_std=True)`:

`Age`, `Income`, `LoanAmount`, `CreditScore`, `MonthsEmployed`, `NumCreditLines`, `InterestRate`, `LoanTerm`, `DTIRatio`

**Categorical Features (7)** — encoded via `OneHotEncoder(drop='first', sparse_output=False)`:

`Education`, `EmploymentType`, `MaritalStatus`, `HasMortgage`, `HasDependents`, `LoanPurpose`, `HasCoSigner`

---

## 5. Risk Score & Transformation Methodology

### Step 1 — Raw Model Output

The Logistic Regression produces a calibrated default probability:

```
P(Default=1 | X) = sigmoid(wᵀ · X + b)    in [0.0000, 1.0000]
```

### Step 2 — Institutional Risk Score

The probability is inverted and scaled to a 0–100 creditworthiness score:

```
risk_score = round( (1.0 - P(Default)) × 100 )    in [0, 100]
```

*Higher scores indicate lower default risk and greater creditworthiness.*

### Step 3 — Threshold Classification

| Risk Level | Probability Range | Risk Score | Classification |
|---|---|---|---|
| **Low Risk** | P < 0.28 | 73 – 100 | No Default (0) |
| **Moderate Risk** | 0.28 ≤ P < 0.50 | 51 – 72 | No Default (0) |
| **High Risk** | P ≥ 0.50 | 0 – 50 | Default (1) |

---

## 6. API Endpoint Documentation

All `/api/*` routes are proxied through Express.js to the Flask ML backend.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Service health check — returns `{"status": "ok", "model": "Logistic Regression"}` |
| `POST` | `/api/predict` | Run full 16-feature inference through preprocessor + production model |
| `GET` | `/api/models` | List all 4 models with summary metrics |
| `GET` | `/api/models/comparison` | 4-model evaluation table, cross-validation metrics, ROC curve data |
| `GET` | `/api/models/:modelId` | Parameters, confusion matrix, and feature coefficients for one model |
| `GET` | `/api/data/insights` | EDA feature distributions, correlation matrix, dataset summary |

### `POST /api/predict` — Example

**Request Body:**
```json
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
```

**Response (`200 OK`):**
```json
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
```

---

## 7. Project Structure

```
LoanLens App/
├── artifacts/                            # ML serialized artifacts
│   ├── metadata/
│   │   ├── model_context.json            # Training config & hyperparameters
│   │   └── selected_model.json           # Production model selection record
│   ├── metrics/
│   │   └── all_metrics.json              # Full 4-model evaluation metrics
│   ├── models/
│   │   ├── logistic_regression.pkl       # Production model
│   │   ├── knn.pkl
│   │   ├── naive_bayes.pkl
│   │   └── decision_tree.pkl
│   └── preprocessing/
│       └── preprocessor.pkl              # Fitted ColumnTransformer pipeline
│
├── data/                                 # Dataset & project metadata
│   ├── Loan_Default.csv                  # Source dataset (255,347 records)
│   └── metadata.json                     # Project-level metadata
│
├── notebooks/                            # Research & training materials
│   ├── Loan_Default_Prediction.ipynb     # Full ML training & evaluation notebook
│   └── Loan_Default_Prediction.pdf       # Exported notebook report
│
├── flask_backend/                        # Python Flask ML inference engine
│   ├── app.py                            # All /api/* routes & model loading
│   └── requirements.txt                  # Python-only dependencies
│
├── public/                               # Static frontend assets
│   ├── css/
│   │   └── app.css                       # Global stylesheet
│   ├── js/                               # Client-side JavaScript
│   │   ├── app.js
│   │   ├── charts.js
│   │   ├── predict.js
│   │   ├── simulator.js
│   │   └── history.js
│   └── artifacts/plots/                  # Pre-generated EDA plot images
│
├── scripts/
│   └── build_artifacts.py                # Re-train & re-serialize .pkl artifacts
│
├── src/                                  # Node.js / TypeScript application
│   ├── lib/                              # Shared internal utilities & data
│   │   ├── constants.ts                  # Navigation registry & global constants
│   │   ├── icons.ts                      # Inline SVG icon library
│   │   ├── metricsData.ts                # Embedded ML evaluation metrics
│   │   ├── mlService.ts                  # Local fallback inference engine
│   │   ├── store.ts                      # In-memory prediction history store
│   │   └── data/
│   │       └── developmentMockData.ts    # Fallback model & dataset definitions
│   ├── services/                         # Business logic service layer
│   │   ├── predictionService.ts          # Prediction proxy & response shaping
│   │   ├── modelService.ts               # Model metrics & comparison
│   │   └── analyticsService.ts           # EDA & insights data
│   └── types/
│       └── index.ts                      # Shared TypeScript type definitions
│
├── views/                                # EJS server-side templates
│   ├── components/                       # Reusable EJS component partials
│   │   ├── model_card.ejs
│   │   └── risk_result_card.ejs
│   ├── pages/                            # Full page templates
│   │   ├── dashboard.ejs
│   │   ├── predict.ejs
│   │   ├── model_context.ejs
│   │   ├── model_comparison.ejs
│   │   ├── model_details.ejs
│   │   ├── data_insights.ejs
│   │   ├── history.ejs
│   │   └── simulator.ejs
│   ├── partials/                         # Layout fragments (header, footer)
│   │   ├── header.ejs
│   │   └── footer.ejs
│   └── 404.ejs
│
├── Dockerfile                            # Unified Node.js + Flask container
├── render.yaml                           # Render Docker deployment config
├── .dockerignore
├── .env.example                          # Environment variable reference
├── requirements.txt                      # Python dependencies (root-level alias)
├── package.json                          # Node.js dependencies & npm scripts
├── server.ts                             # Express.js application entry point
└── tsconfig.json                         # TypeScript compiler config
```
---

## 8. Environment Variables

| Variable | Description | Development Default | Production |
|---|---|---|---|
| `PORT` | Web server listening port | `3000` | Set by cloud provider |
| `HOST` | Web server bind address | `0.0.0.0` | `0.0.0.0` |
| `NODE_ENV` | Runtime environment | `development` | `production` |
| `USE_REMOTE_BACKEND` | Enable Flask proxy routing | `true` | `true` |
| `FLASK_BACKEND_URL` | Target URL for Flask service | `http://127.0.0.1:5001` | `http://127.0.0.1:5001` |
| `FLASK_PORT` | Flask listen port | `5001` | `5001` |
| `FLASK_HOST` | Flask bind address | `127.0.0.1` | `127.0.0.1` |
| `ALLOWED_ORIGINS` | CORS allowed origins for Flask | `*` | Deployed frontend domain |

Copy `.env.example` to `.env` for local development:

```bash
cp .env.example .env
```

---

## 9. Local Development Setup

### Prerequisites

- Node.js v18 or later
- npm v9 or later
- Python 3.10 or later with pip

### Step 1 — Clone & Install

```bash
git clone https://github.com/Bhaviik-5202/LoanLens.git
cd LoanLens

# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2 — Start Development Server

```bash
npm run dev
```

The Express server starts on **http://localhost:3000** and automatically spawns the Flask ML backend on port `5001` as a managed child process. Both services start together — no manual Flask launch required.

### Step 3 — (Optional) Rebuild ML Artifacts

To retrain and re-serialize all `.pkl` artifacts from the source dataset:

```bash
python scripts/build_artifacts.py
```

> **Note:** Requires `Loan_Default.csv` in the project root and scikit-learn 1.6.1 installed.

---

## 10. Production Deployment

### Option A — Docker (Recommended)

The app ships as a single unified container with Node.js and Flask running together under Node process supervision.

```bash
# Build the image
docker build -t loanlens .

# Run the container
docker run -p 3000:3000 \
  -e NODE_ENV=production \
  -e PORT=3000 \
  -e HOST=0.0.0.0 \
  -e USE_REMOTE_BACKEND=true \
  -e FLASK_BACKEND_URL=http://127.0.0.1:5001 \
  loanlens
```

Access the application at **http://localhost:3000**.

### Option B — Render (Current Production)

The repository includes `render.yaml` configured for automatic Docker-based deployment on Render.

1. Push changes to the `main` branch of [github.com/Bhaviik-5202/LoanLens](https://github.com/Bhaviik-5202/LoanLens)
2. Render detects the push, rebuilds the Docker image, and redeploys automatically
3. Health check at `/api/health` must return `200 OK` before traffic is switched

**Live Production URL:** [https://loanlens-hynm.onrender.com](https://loanlens-hynm.onrender.com)

### Option C — Decoupled Multi-Service

Deploy the Node.js frontend and Flask backend as separate cloud microservices:

**Step 1 — Deploy Flask Backend:**
```bash
HOST=0.0.0.0 PORT=5001 python flask_backend/app.py
```

**Step 2 — Deploy Express Frontend:**
```bash
npm run build
USE_REMOTE_BACKEND=true FLASK_BACKEND_URL=https://<your-flask-service-url> npm start
```

---

## 11. Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Web Framework | Express.js | 4.x |
| Language (Backend) | TypeScript | 5.7 |
| Templating | EJS (Server-Side Rendering) | 3.x |
| Styling | Vanilla CSS | — |
| Data Visualization | Chart.js | CDN |
| ML Framework | scikit-learn | 1.6.1 |
| ML API Server | Python Flask + flask-cors | 2.2+ |
| ML Serialization | joblib | 1.2+ |
| Data Processing | pandas, NumPy, SciPy | latest stable |
| Containerization | Docker | — |
| Deployment Platform | Render (Docker runtime) | — |
| Dataset | Kaggle — nikhil1e9/loan-default | 255,347 records |

---

*Built as an end-to-end ML engineering project for academic demonstration of production credit risk modeling.*
