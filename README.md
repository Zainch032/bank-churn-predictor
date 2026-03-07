🏦 **Bank Customer Churn Predictor**

> *"The model scored 80% accuracy and caught zero churners. That's when the real work started."*

End-to-end ML project — EDA, SMOTE, multi-model benchmarking, and a live FastAPI service.
Improved churn **F1-score: 0.50 → 0.62** (+24% relative gain on the metric that actually matters).

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## 💡 The Problem Worth Solving

Banks lose customers silently — no resignation letter, no complaint. By the time it's noticed, they're gone. Acquiring a replacement costs **5–7× more** than retention.

10,000 customers. 14 features. 1 question: **who's about to leave?**

---

## 🏗️ What Was Built

| Phase | Detail |
|---|---|
| 🔍 **EDA** | Crosstabs, boxenplots, pivot tables, heatmaps — every feature interrogated against the target |
| ✂️ **Feature Selection** | 4 features dropped with statistical evidence — `CreditScore`, `EstimatedSalary`, `HasCrCard`, `Tenure` |
| ⚙️ **Preprocessing** | Leak-proof `Pipeline` — `StandardScaler` + `OneHotEncoder` inside `ColumnTransformer` |
| 🤖 **Model Comparison** | 5 classifiers benchmarked on Precision, Recall, F1, ROC-AUC — not accuracy |
| ⚖️ **SMOTE** | Synthetic oversampling on training data only — F1 jumped **0.50 → 0.62** |
| 🚀 **Deployment** | FastAPI REST API + HTML/CSS/JS frontend with prediction form and EDA insights tab |

---

## 🔍 Exploratory Data Analysis

The EDA wasn't just about understanding the data — it was about making every modeling decision defensible. Crosstab analysis on categorical features, boxenplots to compare full distributions (not just means), pivot tables aggregating multiple metrics, and skewness checks on numeric features split by the target.

The goal: understand *who churns* before predicting it.

### What actually separates churners from stayers

- **Age** is the single strongest signal. Churned customers averaged **44.8 years** vs. **37.4** for those who stayed — a 7-year gap that shows up consistently across all geographies and genders.
- **Geography** tells a stark story. Germany sits at **32.4% churn** while France and Spain hover around **16%**. That's double the rate — and German customers also hold the highest average balances (~$120k).
- **Number of products** has the most dramatic non-linear pattern. Two products = **7.6% churn** (loyalty sweet spot). Three products = **82.7%**. Four products = **100%** — every single customer left.
- **Active membership** is the most actionable lever. Inactive members churn at **26.9%** vs. **14.3%** for active ones — nearly double.
- **Balance** is counter-intuitive. Higher-balance customers churn *more* — churned customers averaged **$91k** vs. **$72k** for stayers.

### What doesn't predict churn at all

Just as important:

- **Credit score** (651 stayed vs. 645 churned)
- **Estimated salary** ($100k vs. $101k)
- **Credit card ownership**
- **Tenure**

All showed near-zero difference between groups and were dropped before modeling to reduce noise.

> Full analysis with all visualizations in `notebook/churn.ipynb`

---

## ⚖️ SMOTE — Why It Moved the Needle

80% stayed, 20% churned. Baseline models trained on raw data had a cautious bias — high precision, terrible recall. They only flagged churners when nearly certain, missing most of them. F1 on the churn class: **0.50**.

SMOTE synthesizes new minority-class samples by interpolating between existing ones in feature space — fundamentally different from duplicating rows. Applied inside the pipeline on training data only, no leakage.

| | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Without SMOTE | 0.61 | 0.42 | 0.50 | 0.82 |
| **With SMOTE** | **0.56** | **0.70** | **0.62** | **0.87** |

Lower precision, much higher recall. For retention this is the right trade-off — a false alarm costs a retention email, a missed churner costs a customer.

---

## 🤖 Model Comparison

| Model | F1 (SMOTE) | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.55 | 0.76 |
| K-Nearest Neighbors | 0.52 | 0.74 |
| Decision Tree | 0.53 | 0.72 |
| Random Forest | 0.60 | 0.85 |
| **Gradient Boosting** ✅ | **0.62** | **0.87** |

Gradient Boosting won because churn isn't linear — it captures the `Age × Geography × NumOfProducts` interactions that simpler models miss.

> Full comparison in `notebook/model.ipynb`

---

## 📁 Project Structure

```text
bank-churn-predictor/
│
├── main.py                    # FastAPI application (backend + API)
├── requirements.txt           # Python dependencies
│
├── model/
│   └── model_pipeline.pkl     # Trained sklearn pipeline
│
├── templates/
│   └── index.html             # Frontend UI (Jinja2 template)
│
├── static/
│   ├── style.css              # Modern dashboard styling
│   └── app.js                 # Frontend logic & API calls
│
├── data/
│   ├── Churn_Modelling.csv    # Raw dataset
│   └── Cleanind_data.csv      # Cleaned version
│
└── notebook/
    ├── churn.ipynb            # EDA notebook
    └── model.ipynb            # Modelling, SMOTE, comparison
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/your-username/bank-churn-predictor.git
cd bank-churn-predictor

# (Optional) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # on Windows

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI app
uvicorn main:app --reload
# → http://127.0.0.1:8000  (UI)
# → http://127.0.0.1:8000/docs  (Swagger)
```

---

## 📡 API

```text
POST /predict   →  churn probability + risk level (Low / Medium / High)
GET  /health    →  model load status
GET  /docs      →  Swagger UI
```

**Sample request:**

```json
{
  "CreditScore": 620,
  "Age": 45,
  "Balance": 92000,
  "EstimatedSalary": 72000,
  "NumOfProducts": 1,
  "IsActiveMember": 0,
  "Geography": "Germany",
  "Gender": "Female"
}
```

**Sample response:**

```json
{
  "churn": true,
  "churn_probability": 0.78,
  "risk_level": "High",
  "message": "Target for retention",
  "insights": [
    "Critical Risk: Analysis shows a near 100% exit rate for 4-product users.",
    "Regional Alert: German customers show higher balances and 2x the churn risk.",
    "Demographic Driver: Customer is in the peak churn age bracket (45+)."
  ]
}
```

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Data & ML** | pandas · numpy · scikit-learn · imbalanced-learn |
| **Visualization** | matplotlib · seaborn |
| **Backend** | FastAPI · Uvicorn · Pydantic · Jinja2 |
| **Frontend** | HTML5 · CSS3 · Vanilla JS |

---

## Skills

`EDA` · `Feature Selection` · `Class Imbalance` · `SMOTE` · `sklearn Pipelines` · `ColumnTransformer` · `Model Evaluation` · `F1-Score` · `ROC-AUC` · `Gradient Boosting` · `Random Forest` · `FastAPI` · `REST API` · `End-to-End ML`

---

MIT License

