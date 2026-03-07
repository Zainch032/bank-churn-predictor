# 🏦 Bank Customer Churn Predictor

**End-to-end machine learning project** — from raw data exploration to a live prediction API. Built to solve a real business problem: identifying bank customers likely to leave before they do.

> **Highlight:** Improved churn detection F1-score from **0.50 → 0.62** by diagnosing class imbalance and applying SMOTE — a 24% relative gain on the metric that actually matters.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## The Problem

Retail banks lose customers silently. By the time someone closes their account, it's too late — and acquiring a replacement costs 5–7× more than keeping the original customer. The question this project answers is simple: **can we predict who's about to leave, early enough to do something about it?**

The dataset has 10,000 customers, 14 features, and a 20% churn rate. That imbalance is the central challenge — and ignoring it is the most common mistake people make on this type of problem.

---

## What Was Built

A complete ML pipeline from scratch:

- **Deep exploratory analysis** to find which features actually separate churners from stayers — and just as importantly, which ones don't
- **Feature selection** based on statistical evidence, not assumptions
- **Preprocessing pipeline** with scaling and encoding, structured to prevent data leakage
- **Five classifiers benchmarked** side-by-side on the right metrics
- **SMOTE** applied to fix the imbalance problem — the single change that moved the needle most
- **REST API** built with FastAPI, serving predictions with probability scores and risk levels
- **Interactive frontend** with a predict form and an EDA insights dashboard — no frameworks, pure HTML/CSS/JS

---

## The Data Science Story

### Starting Point — Exploratory Data Analysis

Before touching a model, the full dataset was interrogated through EDA. This meant going beyond simple histograms: normalized crosstabs to measure churn rates per category, boxenplots to compare full distributions between churned and stayed groups, pivot tables aggregating multiple metrics across geography and exit status simultaneously, and skewness analysis across every numeric feature split by the target.

The goal was to build a mental model of *who churns and why* — and to make feature selection a data-driven decision rather than a guess.

**The most important findings:**

— Age is the strongest signal. Customers who churned averaged 44.8 years old vs. 37.4 for those who stayed. No other feature had this level of distributional separation.

— Germany has double the churn rate of France and Spain (32% vs. ~16%), and German customers also hold the highest average balances. Highest risk, highest value — a critical retention priority.

— The product count relationship is non-linear and dramatic. Two products is the loyalty sweet spot at 7.6% churn. Three products jumps to 82.7%. Four products: 100% — every single customer with four products exited. This kind of pattern only surfaces if you look at churn rate *per value* rather than aggregate statistics.

— Inactive members churn at nearly twice the rate of active ones. Engagement level is a more actionable lever than almost any financial metric.

— Counter-intuitively, higher balance customers churn more. The bank is losing its wealthiest segment — the opposite of what you'd expect and exactly the kind of insight that justifies doing EDA properly.

**What got dropped:** Credit score (651 vs. 645 across groups — statistically meaningless), estimated salary (virtually identical between groups), credit card ownership (no churn difference), and tenure (near-identical distributions across all years). Including these would have added noise, not signal.

---

### The Imbalance Problem

80% stayed, 20% churned. A model that predicts "stay" for every customer scores 80% accuracy and catches zero churners. Accuracy is the wrong metric entirely for this problem.

The baseline models — trained on the raw imbalanced data — hit an F1-score of **0.50** on the churn class. High precision, terrible recall. The model was cautious about labeling anyone as a churner, so it missed most of the actual ones.

**SMOTE (Synthetic Minority Over-sampling Technique)** generates synthetic minority-class samples by interpolating between existing ones in feature space — it's fundamentally different from just duplicating rows. Applied correctly inside the training pipeline (never touching the test set), it rebalanced the training distribution and forced the model to genuinely learn the minority class.

The result: recall on churners jumped significantly, F1 moved from **0.50 → 0.62**. The model now catches more actual churners — which is the right trade-off, because in retention, missing a churner is far more costly than a false alarm.

---

### Model Comparison

Five models were trained and evaluated — Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and Gradient Boosting — both with and without SMOTE. Every model was compared on **Precision, Recall, F1, and ROC-AUC** on the minority class, not overall accuracy.

Gradient Boosting with SMOTE won on both F1 (0.62) and ROC-AUC (0.87). It handles the non-linear feature interactions in this dataset — Age × Geography × NumOfProducts — that simpler models can't capture. The trained pipeline (preprocessor + model) was serialized and deployed directly.

---

### Deployment

The final pipeline is served through a **FastAPI** backend with automatic Swagger documentation. The frontend has two sections: a prediction form where you enter customer details and get back a churn probability with a risk level, and an EDA insights tab summarizing the key findings from the analysis. Static assets are properly separated (HTML / CSS / JS), and the model loads once at startup.

---

## Skills Demonstrated

`Exploratory Data Analysis` · `Statistical Feature Selection` · `Class Imbalance` · `SMOTE` · `scikit-learn Pipelines` · `ColumnTransformer` · `StandardScaler` · `OneHotEncoder` · `Logistic Regression` · `Random Forest` · `Gradient Boosting` · `Model Evaluation` · `F1-Score` · `ROC-AUC` · `Precision-Recall Trade-off` · `FastAPI` · `REST API Design` · `Jinja2` · `HTML/CSS/JS` · `End-to-End ML`

---

## Project Structure

```
bank-churn-predictor/
├── app/
│   ├── main.py              FastAPI app
│   ├── requirements.txt
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── app.js
├── data/
│   ├── Churn_Modelling.csv
│   └── Cleanind_data.csv
└── notebook/
    ├── churn.ipynb
    └── model.ipynb
```

---

## Quick Start

```bash
git clone https://github.com/your-username/bank-churn-predictor.git
cd bank-churn-predictor
pip install -r app/requirements.txt
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

---

## Tech Stack

| | |
|---|---|
| Data & ML | Python, pandas, numpy, scikit-learn, imbalanced-learn |
| Visualization | matplotlib, seaborn |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | HTML5, CSS3, Vanilla JS |

 

---

## License

MIT
