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
2. [Key Features](#2-key-features)
3. [Machine Learning Models & Evaluation](#3-machine-learning-models--evaluation)
4. [Dataset & Preprocessing](#4-dataset--preprocessing)
5. [Risk Scoring & Mathematical Methodology](#5-risk-scoring--mathematical-methodology)
6. [API Endpoint Documentation & Examples](#6-api-endpoint-documentation--examples)
7. [Project Structure](#7-project-structure)
8. [Environment Variables](#8-environment-variables)
9. [Local Development Setup](#9-local-development-setup)
10. [Production Deployment](#10-production-deployment)
11. [Tech Stack](#11-tech-stack)

---

## 1. Architecture

LoanLens uses a decoupled two-tier architecture running inside a single container:

- **Frontend & Web Layer**: Express.js (Node.js / TypeScript) serving server-side rendered (SSR) EJS views, styled with Vanilla CSS and interactive Chart.js charts.
- **ML Inference Engine**: Python Flask service loading pre-trained scikit-learn pipelines and serialized `.pkl` models once at startup into memory.
- **Inter-Service Communication**: Node.js proxies API requests (`/api/*`) to Flask over localhost or remote backend URL with automated health checking and fallback support.

```
Browser Client
      │
      │ HTTP :3000
      ▼
Express.js (TypeScript) ────── EJS SSR Pages & Static Assets
      │
      │ /api/predict
      │ /api/models
      │ /api/data/insights
      ▼
Flask ML Engine ────────────── HTTP :5001
      │
      ▼
Scikit-Learn Artifacts
  ├── artifacts/preprocessing/preprocessor.pkl
  ├── artifacts/models/logistic_regression.pkl   <-- Production Model
  ├── artifacts/models/knn.pkl
  ├── artifacts/models/naive_bayes.pkl
  └── artifacts/models/decision_tree.pkl
```

---

## 2. Key Features

| Feature | Description |
|---|---|
| **Interactive Risk Assessment** | Comprehensive 16-attribute borrower evaluation scored via L2 Logistic Regression with instant default probability and confidence metrics. |
| **Institutional Risk Score (0–100)** | Calibrated creditworthiness score with color-coded risk bands (Prime, Near Prime, Subprime, High Risk). |
| **Explainable Factor Attribution** | Per-prediction log-odds coefficient breakdown identifying top protective mitigants and risk-amplifying attributes. |
| **Multi-Model Benchmark Matrix** | Comprehensive comparative analysis of 4 classification algorithms on identical 5-fold CV splits, ROC curves, and confusion matrices. |
| **Dynamic Sensitivity Simulator** | Real-time parameter perturbation engine for what-if sensitivity analysis across credit scores, DTI, income, and loan amounts. |
| **Exploratory Data Analysis (EDA)** | Interactive data explorer featuring correlation heatmaps, feature distribution histograms, box plots, and categorical default rates. |
| **Prediction History Store** | In-memory session store tracking historical assessments with reloadable profile inspection. |

---

## 3. Machine Learning Models & Evaluation

LoanLens evaluates four primary classification algorithms on the 255k dataset, along with a from-scratch reference implementation:

| Model ID | Algorithm | Scikit-Learn Class | ROC-AUC | 5-Fold CV AUC | Default Recall | F1-Score | Status |
|---|---|---|---|---|---|---|---|
| **logistic-regression** | L2-Regularized Logistic Regression (Balanced) | `LogisticRegression` | **0.7532** | **0.7426 ± 0.006** | **69.95%** | **0.3753** | **✅ Production Model** |
| **knn** | Distance-Weighted K-Nearest Neighbors (k=25) | `KNeighborsClassifier` | 0.6927 | 0.6753 ± 0.007 | 0.70% | 0.0137 | Benchmark |
| **naive-bayes** | Gaussian Naive Bayes (`var_smoothing=1e-5`) | `GaussianNB` | 0.7499 | 0.7396 ± 0.004 | 2.56% | 0.0487 | Benchmark |
| **decision-tree** | Entropy-Partitioned Decision Tree (`max_depth=6`) | `DecisionTreeClassifier` | 0.7254 | 0.7036 ± 0.006 | 3.02% | 0.0573 | Benchmark |
| *scratch-logistic* | Batch Gradient Descent (Pure NumPy) | `NumPy` | 0.7530 | 0.7530 ± 0.003 | 68.40% | 0.3690 | Reference |

### Why Logistic Regression is the Selected Model

In credit underwriting, **false negatives** (approving an applicant who ultimately defaults) are significantly more expensive than false positives. The balanced Logistic Regression model was selected because:

1. **Superior Default Recall (69.95%)**: Correctly identifies 4,149 out of 5,931 test defaults under heavy class imbalance (11.6% default rate).
2. **Highest Discriminative Power (0.7532 ROC-AUC)**: Outperforms tree-based and distance-based baselines across all classification thresholds.
3. **Regulatory Explainability**: Standardized log-odds coefficients ($w_i$) enable direct, transparent factor attribution for auditability and compliance.

### Top Predictive Features (Log-Odds Impact)

- **Risk Amplifiers (+ Log-Odds)**: High Interest Rate ($+0.72$), High DTI Ratio ($+0.45$), Extended Loan Term ($+0.31$), Unemployed Employment Type ($+0.28$).
- **Protective Mitigants (- Log-Odds)**: Higher Credit Score ($-0.68$), High Annual Income ($-0.54$), Long Employment Duration ($-0.39$), Co-Signer Present ($-0.22$).

---

## 4. Dataset & Preprocessing

| Property | Value |
|---|---|
| **Source** | Kaggle — `nikhil1e9/loan-default` |
| **Total Observations** | 255,347 clean consumer records |
| **Duplicates / Missing Values** | 0 / 0 |
| **Class Distribution** | Non-Default: 225,694 (88.39%) \| Default: 29,653 (11.61%) |
| **Raw Input Features** | 16 features (9 numerical, 7 categorical) |
| **Transformed Feature Space** | 24 continuous dimensions post-encoding |

### Transformation Pipeline (`artifacts/preprocessing/preprocessor.pkl`)

- **Numerical Standardization (`StandardScaler`)**:
  Applied to `Age`, `Income`, `LoanAmount`, `CreditScore`, `MonthsEmployed`, `NumCreditLines`, `InterestRate`, `LoanTerm`, `DTIRatio`.
- **Categorical Encoding (`OneHotEncoder(drop='first')`)**:
  Applied to `Education`, `EmploymentType`, `MaritalStatus`, `HasMortgage`, `HasDependents`, `LoanPurpose`, `HasCoSigner`.

---

## 5. Risk Scoring & Mathematical Methodology

### Step 1 — Calibrated Default Probability

The Logistic Regression model computes the posterior probability of default using the standard sigmoid link:

$$P(	ext{Default}=1 \mid X) = \sigma(w^T X + b) = rac{1}{1 + e^{-(w^T X + b)}}$$

### Step 2 — Institutional Risk Score (0–100)

The probability of default is inverted to produce an institutional creditworthiness score where higher numbers indicate superior credit quality:

$$	ext{RiskScore} = 	ext{round}\Big((1.0 - P(	ext{Default})) 	imes 100\Big) \in [0, 100]$$

| Risk Score Band | Risk Level | Model Classification | Typical Underwriting Guidance |
|---|---|---|---|
| **80 – 100** | Low Risk | No Default | Standard automated prime approval |
| **60 – 79** | Moderate Risk | Borderline / Low Default | Standard approval with income verification |
| **40 – 59** | Elevated Risk | Moderate Default Risk | Secondary underwriting review / collateral check |
| **0 – 39** | High Risk | Default | Decline or require qualified co-signer |

---

## 6. API Endpoint Documentation & Examples

All API routes are served at `/api/*` and return standard JSON payloads.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health status and active inference model |
| `POST` | `/api/predict` | Full 16-feature loan risk inference |
| `POST` | `/api/simulate` | Parameter sensitivity sweep across risk factors |
| `GET` | `/api/models` | Metadata and metrics for all 4 benchmark models |
| `GET` | `/api/models/comparison` | Full comparison matrix, CV statistics, and ROC data |
| `GET` | `/api/models/:modelId` | Granular parameters, confusion matrix, and feature importances |
| `GET` | `/api/data/insights` | EDA distributions, correlation matrix, and summary stats |

### Example: Running Risk Prediction (`POST /api/predict`)

**cURL Request:**
```bash
curl -X POST https://loanlens-hynm.onrender.com/api/predict   -H "Content-Type: application/json"   -d '{
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
  }'
```

**Response (`200 OK`):**
```json
{
  "success": true,
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
      "Solid financial fundamentals serve as strong protective mitigants."
    ]
  }
}
```

---

## 7. Project Structure

```
LoanLens App/
├── artifacts/                            # Serialized ML assets & benchmarks
│   ├── metadata/
│   │   ├── model_context.json            # Hyperparameters & dataset stats
│   │   └── selected_model.json           # Active production model details
│   ├── models/                           # Serialized scikit-learn models (.pkl)
│   │   ├── logistic_regression.pkl       # Production model (Balanced L2)
│   │   ├── knn.pkl                       # Benchmark KNN
│   │   ├── naive_bayes.pkl               # Benchmark Naive Bayes
│   │   └── decision_tree.pkl             # Benchmark Decision Tree
│   └── preprocessing/
│       └── preprocessor.pkl              # Fitted ColumnTransformer pipeline
│
├── data/                                 # Datasets & project metadata
│   ├── Loan_Default.csv                  # Source dataset (255,347 records)
│   └── metadata.json                     # Project schema & field definitions
│
├── notebooks/                            # Jupyter training & research notebooks
│   ├── Loan_Default_Prediction.ipynb     # Full EDA, training, and evaluation pipeline
│   └── Loan_Default_Prediction.pdf       # Exported research report
│
├── flask_backend/                        # Python Flask ML microservice
│   ├── app.py                            # Flask API routes, inference & artifact loader
│   └── requirements.txt                  # Python dependencies
│
├── public/                               # Static assets served by Express
│   ├── css/
│   │   └── app.css                       # Global stylesheet
│   ├── js/                               # Frontend controller scripts
│   │   ├── app.js
│   │   ├── charts.js
│   │   ├── predict.js
│   │   ├── simulator.js
│   │   └── history.js
│   └── artifacts/plots/                  # Pre-rendered high-res EDA visualizations
│
├── scripts/
│   └── build_artifacts.py                # Automated training & artifact export script
│
├── src/                                  # Node.js / TypeScript application layer
│   ├── lib/                              # Core utilities, constants & embedded data
│   │   ├── constants.ts                  # Navigation registry & global constants
│   │   ├── icons.ts                      # Inline SVG icon library
│   │   ├── metricsData.ts                # Embedded benchmark metrics
│   │   ├── mlService.ts                  # Local inference & payload normalization
│   │   ├── store.ts                      # In-memory prediction history store
│   │   └── data/
│   │       └── developmentMockData.ts    # Fallback model specifications
│   ├── services/                         # Business logic & proxy services
│   │   ├── predictionService.ts          # Prediction proxy & response formatting
│   │   ├── modelService.ts               # Model catalog & comparison aggregator
│   │   └── analyticsService.ts           # EDA insights service
│   └── types/
│       └── index.ts                      # TypeScript type definitions
│
├── views/                                # Server-side EJS templates
│   ├── components/                       # Reusable UI components
│   │   ├── model_card.ejs
│   │   └── risk_result_card.ejs
│   ├── pages/                            # Full-page templates
│   │   ├── dashboard.ejs                 # Overview dashboard
│   │   ├── predict.ejs                   # New assessment form
│   │   ├── model_context.ejs             # Mathematical context & problem formulation
│   │   ├── model_comparison.ejs          # 4-model evaluation benchmark
│   │   ├── model_details.ejs             # Deep inspection of model hyperparameters
│   │   ├── data_insights.ejs             # EDA visual explorer
│   │   ├── history.ejs                   # Assessment audit log
│   │   └── simulator.ejs                 # Sensitivity analysis tool
│   ├── partials/                         # Layout headers & footers
│   │   ├── header.ejs
│   │   └── footer.ejs
│   └── 404.ejs
│
├── Dockerfile                            # Production multi-stage Docker container
├── render.yaml                           # Infrastructure as code for Render deployment
├── .dockerignore
├── .gitignore
├── .env.example                          # Environment variable configuration template
├── package.json                          # Node dependencies & npm scripts
├── server.ts                             # Production Express.js server entry point
└── tsconfig.json                         # TypeScript compiler configuration
```

---

## 8. Environment Variables

| Variable | Description | Development Default | Production Default |
|---|---|---|---|
| `PORT` | Node.js web server port | `3000` | Injected by hosting platform |
| `HOST` | Node.js bind address | `0.0.0.0` | `0.0.0.0` |
| `NODE_ENV` | Application environment | `development` | `production` |
| `USE_REMOTE_BACKEND` | Enable Flask backend routing | `true` | `true` |
| `FLASK_BACKEND_URL` | Target Flask service address | `http://127.0.0.1:5001` | `http://127.0.0.1:5001` |
| `FLASK_PORT` | Python Flask listening port | `5001` | `5001` |
| `FLASK_HOST` | Python Flask bind host | `127.0.0.1` | `127.0.0.1` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `*` | Deployed domain |

---

## 9. Local Development Setup

### Prerequisites

- **Node.js**: v18.0.0 or higher
- **npm**: v9.0.0 or higher
- **Python**: 3.10 or higher with `pip`

### Step 1 — Clone Repository & Install Dependencies

```bash
git clone https://github.com/Bhaviik-5202/LoanLens.git
cd LoanLens

# Install Node.js dependencies
npm install

# Install Python ML dependencies
pip install -r flask_backend/requirements.txt
```

### Step 2 — Start Full-Stack Development Environment

```bash
npm run dev
```

> **Automated Process Management**: Express will automatically start on `http://localhost:3000` and seamlessly spawn the Python Flask inference engine on port `5001` in the background.

### Step 3 — Available NPM Scripts

| Script | Command | Description |
|---|---|---|
| `npm run dev` | `tsx server.ts` | Start live development server with hot-reload |
| `npm run build` | `tsc` | Compile TypeScript source code to `dist/` |
| `npm start` | `node dist/server.js` | Run compiled production server |
| `npm run lint` | `tsc --noEmit` | Run static type checking without generating build output |

---

## 10. Production Deployment

### Option A — Unified Docker Deployment (Recommended)

The application is containerized with a unified Docker container running both the Node.js Express server and the Python Flask inference service.

```bash
# Build the Docker image
docker build -t loanlens .

# Run the container locally
docker run -p 3000:3000   -e NODE_ENV=production   -e PORT=3000   -e USE_REMOTE_BACKEND=true   loanlens
```

Visit **http://localhost:3000** to access the application.

### Option B — Render Cloud Deployment

The repository is configured for direct continuous deployment on Render via `render.yaml`.

1. Connect the GitHub repository `Bhaviik-5202/LoanLens` to Render.
2. Render detects `render.yaml` and deploys using the Docker environment.
3. Health check at `/api/health` validates container availability before switching traffic.

**Live Application URL:** [https://loanlens-hynm.onrender.com](https://loanlens-hynm.onrender.com)

---

## 11. Tech Stack

| Layer | Technology | Specification / Version |
|---|---|---|
| **Web Server** | Express.js | 4.21+ |
| **Language (Web)** | TypeScript | 5.7+ |
| **Templating Engine** | EJS (SSR) | 3.1+ |
| **Styling** | Vanilla CSS3 | Modern CSS custom properties & glassmorphism |
| **Visualizations** | Chart.js | 4.4.1 |
| **ML Inference API** | Python Flask | 2.2+ |
| **Machine Learning** | scikit-learn | 1.6.1 |
| **Data Processing** | pandas, NumPy, SciPy | Latest stable |
| **Model Serialization**| joblib | 1.2+ |
| **Containerization** | Docker | Multi-stage build (`python:3.11-slim` + Node.js 20) |
| **Cloud Hosting** | Render | Docker runtime |
| **Dataset** | Kaggle | 255,347 borrower records (`nikhil1e9/loan-default`) |

---

## License & Credits

- **Author**: Bhavik Parmar
- **Academic Context**: Machine Learning (CSE Sem-5) Institutional Risk Assessment Demonstration
- **Dataset Attribution**: Kaggle — `nikhil1e9/loan-default`
