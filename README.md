🏦 **Bank Customer Churn Predictor**

> *"The model scored 80% accuracy and caught zero churners. That's when the real work started."*

End-to-end ML project — EDA, SMOTE, multi-model benchmarking, and a live FastAPI service.
Improved churn **F1-score: 0.50 → 0.62** (+24% relative gain on the metric that actually matters).

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## 🚀 Live Demo

Try the deployed app here: [Bank Churn – Hugging Face Space](https://huggingface.co/spaces/Zainch032/Bank_Churn)

---

## 💡 The Problem Worth Solving

Banks lose customers silently — no resignation letter, no complaint. By the time it's noticed, they're gone. Acquiring a replacement costs **5–7× more** than retention.

10,000 customers. 14 features. 1 question: **who's about to leave?**

---

## 🏗️ What Was Built

| Phase | Detail |
|---|---|
| 🔍 **EDA** | Crosstabs, boxenplots, pivot tables, heatmaps — every feature interrogated against the target |
| ✂️ **Feature Selection** | 4 features dropped with statistical evidence |
| ⚙️ **Preprocessing** | Leak-proof `Pipeline` — `StandardScaler` + `OneHotEncoder` inside `ColumnTransformer` |
| 🤖 **Model Comparison** | 5 classifiers benchmarked on Precision, Recall, F1, ROC-AUC — not accuracy |
| ⚖️ **SMOTE** | Synthetic oversampling on training data only — F1 jumped **0.50 → 0.62** |
| 🚀 **Deployment** | FastAPI REST API + HTML/CSS/JS frontend with prediction form and EDA insights tab |

---

## 🔍 Exploratory Data Analysis

The EDA wasn't just about understanding data — every finding had to justify a modeling decision. No chart was made for decoration.

### Dataset Overview
- **10,000 customers** · **14 features** · **20.4% churn rate**
- Tools used: crosstab analysis, boxenplots, pivot tables, correlation heatmaps, skewness checks

---

### Age — The Strongest Signal
Age was the single most predictive feature across every slice of the data.

| Group | Average Age |
|---|---|
| Stayed | 37.4 years |
| Churned | 44.8 years |

A **7-year gap** that held consistent across all geographies and genders. Customers in the 45–60 age bracket were dramatically overrepresented in churners. Younger customers showed strong loyalty regardless of other factors.

The pattern was not just a mean difference — the entire age distribution shifted. Boxenplots showed the upper tail of the churned group extending well into the 55–65 range while stayers clustered tightly around 30–40.

**Modeling implication:** Age was kept as a top feature and confirmed via feature importance after training. It ranked as the most important variable in the final Gradient Boosting model.

---

### Geography — Double the Risk in Germany
Geography turned out to be far more than a demographic label.

| Country | Churn Rate | Avg Balance |
|---|---|---|
| France | ~16% | ~$62k |
| Spain | ~16% | ~$61k |
| **Germany** | **32.4%** | **~$120k** |

Germany has **double the churn rate** of France and Spain — and German customers also hold the highest average balances. This is a critical business insight: the most valuable customers by balance are also the most likely to leave.

France and Spain behaved almost identically across every metric — suggesting geography encodes something structural about the banking relationship or market competition in each country rather than just demographics.

**Modeling implication:** Geography was kept as a categorical feature. One-hot encoding was applied inside the pipeline to avoid ordinal assumptions.

---

### Number of Products — The Most Dramatic Pattern
This was the most surprising and non-linear finding in the entire dataset.

| Products | Churn Rate |
|---|---|
| 1 product | 27.7% |
| **2 products** | **7.6%** ← loyalty sweet spot |
| 3 products | 82.7% |
| 4 products | 100% |

Two products is the loyalty sweet spot — customers feel engaged without feeling trapped. Three or four products shows a complete reversal — every single 4-product customer churned. This suggests over-selling or forced bundling creates resentment rather than loyalty.

The jump from 7.6% to 82.7% between two and three products is one of the most dramatic non-linear relationships possible in a dataset. No linear model can capture this — it requires splits and interactions.

**Modeling implication:** This non-linear relationship is exactly why tree-based models outperformed linear ones. Gradient Boosting captured this interaction naturally through its sequential split structure.

---

### Active Membership — Most Actionable Lever
Active membership status was the most directly actionable finding for a retention team.

| Status | Churn Rate |
|---|---|
| Active member | 14.3% |
| Inactive member | 26.9% |

Inactive members churn at nearly **double the rate**. Unlike age or geography which cannot be changed, engagement can be directly influenced through targeted campaigns, app notifications, rewards programs, and personalized outreach.

This makes active membership the highest-ROI intervention point — a customer flagged as inactive AND high-risk by the model is the ideal target for a retention campaign before they decide to leave.

**Modeling implication:** Kept as a binary feature. Ranked as second most important variable in the final model after age.

---

### Balance — Counter-Intuitive Finding
Higher balance customers churn *more* — opposite of what most banks assume.

| Group | Average Balance |
|---|---|
| Stayed | ~$72k |
| Churned | ~$91k |

Wealthier customers are more likely to have multiple banking relationships and are more financially sophisticated — they actively compare rates, fees, and services across providers. A better offer from a competitor is more likely to move them.

This also partially explains the Germany pattern — German customers had both the highest average balances and the highest churn rate.

**Modeling implication:** Balance kept as a numeric feature with standard scaling applied inside the pipeline.

---

### Gender — Moderate Signal
Female customers churned at a noticeably higher rate than male customers.

| Gender | Churn Rate |
|---|---|
| Male | ~16% |
| Female | ~25% |

The gap held across geographies suggesting it reflects something about product-market fit or service experience rather than just demographic composition of the sample.

**Modeling implication:** Kept as a binary encoded feature.

---

### Features Dropped — With Evidence

These 4 features were dropped before modeling — not arbitrarily, but because the data showed near-zero difference between churners and stayers:

| Feature | Stayed Avg | Churned Avg | Decision |
|---|---|---|---|
| CreditScore | 651 | 645 | Dropped |
| EstimatedSalary | $100k | $101k | Dropped |
| HasCrCard | 70.5% yes | 70.9% yes | Dropped |
| Tenure | 5.03 years | 4.93 years | Dropped |

Dropping noisy features reduced model complexity, shortened training time, and slightly improved generalization by removing features that added variance without signal.

> Full visualizations and code in `notebook/churn.ipynb`

---

## ⚖️ SMOTE — Why It Moved the Needle

80% stayed, 20% churned. Baseline models trained on raw data had a cautious bias — high precision, terrible recall. They only flagged churners when nearly certain, missing most of them. F1 on the churn class: **0.50**.

SMOTE synthesizes new minority-class samples by interpolating between existing ones in feature space — fundamentally different from simply duplicating rows. It creates plausible new examples that fill in the decision boundary rather than just repeating existing points.

Applied inside the pipeline on training data only — no leakage into validation or test sets.

| | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Without SMOTE | 0.61 | 0.42 | 0.50 | 0.82 |
| **With SMOTE** | **0.56** | **0.70** | **0.62** | **0.87** |

Lower precision, much higher recall. For retention this is the right trade-off — a false alarm costs a retention email, a missed churner costs a customer worth thousands in lifetime value.

---

## 🤖 Model Comparison

| Model | F1 (SMOTE) | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.55 | 0.76 |
| K-Nearest Neighbors | 0.52 | 0.74 |
| Decision Tree | 0.53 | 0.72 |
| Random Forest | 0.60 | 0.85 |
| **Gradient Boosting** ✅ | **0.62** | **0.87** |

Gradient Boosting won because churn isn't linear — it captures the `Age × Geography × NumOfProducts` interactions that simpler models miss. Each tree corrects the errors of the previous one, gradually learning the complex boundary between churners and stayers.

Logistic Regression confirmed the feature selection decisions were sound — even a linear model reached 0.55 F1, meaning the remaining features carried real signal. The gap between Logistic Regression and Gradient Boosting (0.55 → 0.62) represents the value of capturing non-linear interactions.

> Full comparison in `notebook/model.ipynb`

---

## 📁 Project Structure

```text
bank-churn-predictor/
├── main.py
├── requirements.txt
├── model/
│   └── model_pipeline.pkl
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── app.js
├── data/
│   ├── Churn_Modelling.csv
│   └── Cleanind_data.csv
└── notebook/
    ├── churn.ipynb
    └── model.ipynb
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/Zainch032/Bank_Churn.git
cd Bank_Churn
pip install -r requirements.txt
uvicorn main:app --reload
# → http://127.0.0.1:8000       (UI)
# → http://127.0.0.1:8000/docs  (Swagger)
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

## Author

**Muhammad Zain** — Final year Data Science student, Lahore, Pakistan

- GitHub: [Zainch032](https://github.com/Zainch032)
- LinkedIn: [Muhammad Zain](https://linkedin.com/in/muhammad-zain-9710692b4)
- Hugging Face: [Zainch032](https://huggingface.co/Zainch032)

---

MIT License
```

---

