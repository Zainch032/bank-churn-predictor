# 🏦 Bank Customer Churn Predictor

> *"The model scored 80% accuracy and caught zero churners. That's when the real work started."*

End-to-end ML project — EDA, SMOTE, multi-model benchmarking, and a live FastAPI service.
Improved churn **F1-score: 0.50 → 0.62** (+24% relative gain on the metric that actually matters).

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
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
| 🔍 EDA | Crosstabs, boxenplots, pivot tables, heatmaps — every feature interrogated against the target |
| ✂️ Feature Selection | 4 features dropped with statistical evidence — `CreditScore`, `EstimatedSalary`, `HasCrCard`, `Tenure` |
| ⚙️ Preprocessing | Leak-proof `Pipeline` — `StandardScaler` + `OneHotEncoder` inside `ColumnTransformer` |
| 🤖 Model Comparison | 5 classifiers benchmarked on Precision, Recall, F1, ROC-AUC — not accuracy |
| ⚖️ SMOTE | Synthetic oversampling on training data only — F1 jumped **0.50 → 0.62** |
| 🚀 Deployment | FastAPI REST API + HTML/CSS/JS frontend with prediction form and EDA insights tab |

---

## 🔍 What the Data Said

Ran deep EDA across `churn.ipynb` before touching any model. The interesting part wasn't what predicted churn — it was what *didn't*.

**Credit score, salary, tenure, credit card ownership** — four features that sound important — showed near-zero difference between churners and stayers. Dropped all four.

What remained: age, geography, product count, balance, activity status. Five features carrying almost all the signal.

> Full analysis in `notebook/churn.ipynb`

---

## 😮 3 Findings That Were Actually Surprising

**High-balance customers churn MORE.**
Churned customers averaged $91k vs. $72k for stayers. The bank is losing its wealthiest segment — counterintuitive and worth flagging as a strategy problem.

**4 products = 100% churn rate.**
Every customer with 4 products left. 2 products? 7.6% — the loyalty sweet spot. Completely invisible in a correlation matrix.

**Germany is a country-sized red flag.**
32% churn vs. ~16% in France and Spain — double — combined with the highest average balances (~$120k). Highest risk and highest value in the same segment.

---

## ⚖️ SMOTE — Why It Moved the Needle

Baseline models had a cautious bias — high precision, terrible recall. They only flagged churners when nearly certain, so they missed most of them.

SMOTE synthesizes minority-class samples by interpolating in feature space (not duplicating rows), applied inside the pipeline on training data only — no leakage.

| | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Without SMOTE | 0.61 | 0.42 | 0.50 | 0.82 |
| **With SMOTE** | **0.56** | **0.70** | **0.62** | **0.87** |

Lower precision, much higher recall. In retention that's the right trade-off — a false alarm costs a retention email, a missed churner costs a customer.

---

## 🤖 Model Comparison

| Model | F1 (SMOTE) | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.55 | 0.76 |
| K-Nearest Neighbors | 0.52 | 0.74 |
| Decision Tree | 0.53 | 0.72 |
| Random Forest | 0.60 | 0.85 |
| **Gradient Boosting** ✅ | **0.62** | **0.87** |

> Full comparison in `notebook/model.ipynb`

---

## 📁 Project Structure

```
bank-churn-predictor/
│
├── app/
│   ├── main.py                  FastAPI application
│   ├── requirements.txt
│   ├── model/
│   │   └── model_pipeline.pkl   Trained sklearn pipeline
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── app.js
│
├── data/
│   ├── Churn_Modelling.csv      Raw dataset
│   └── Cleanind_data.csv        Cleaned version
│
└── notebook/
    ├── churn.ipynb              EDA notebook
    └── model.ipynb              Modelling, SMOTE, comparison
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/your-username/bank-churn-predictor.git
cd bank-churn-predictor
pip install -r app/requirements.txt
uvicorn app.main:app --reload
# → http://127.0.0.1:8000
```

---

## 📡 API

```
POST /predict   →  churn probability + risk level (Low / Medium / High)
GET  /health    →  model load status
GET  /docs      →  Swagger UI
```

**Sample request:**
```json
{ "Age": 45, "Balance": 92000, "Geography": "Germany",
  "Gender": "Female", "NumOfProducts": 1, "IsActiveMember": 0,
  "CreditScore": 620, "EstimatedSalary": 72000 }
```

**Sample response:**
```json
{ "churn": true, "churn_probability": 0.78, "risk_level": "High",
  "message": "Customer is likely to churn." }
```

---

## 🛠 Tech Stack

| | |
|---|---|
| Data & ML | pandas · numpy · scikit-learn · imbalanced-learn |
| Visualization | matplotlib · seaborn |
| Backend | FastAPI · Uvicorn · Pydantic · Jinja2 |
| Frontend | HTML5 · CSS3 · Vanilla JS |

---

## Skills

`EDA` · `Feature Selection` · `Class Imbalance` · `SMOTE` · `sklearn Pipelines` · `ColumnTransformer` · `Model Evaluation` · `F1-Score` · `ROC-AUC` · `Gradient Boosting` · `Random Forest` · `FastAPI` · `REST API` · `End-to-End ML`

---

*MIT License*
