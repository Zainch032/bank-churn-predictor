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

## 🔍 How Patterns Were Discovered

The EDA followed a **systematic, hypothesis-driven approach** — every visualization tested a specific question about churn behavior. No chart was made for decoration.

### Discovery Methodology

**Step 1: Distribution Analysis**
- Examined class balance (20.4% churn rate) to understand baseline and plan SMOTE strategy
- Used skewness checks to identify non-normal distributions requiring transformation

**Step 2: Univariate Feature Analysis**
- **Crosstab heatmaps** for categorical features (Gender, Geography, IsActiveMember) — normalized to show churn rates per group
- **Boxenplots** for numeric features (Age, Balance, CreditScore) — better than boxplots for showing distribution tails and outliers
- **Pivot tables** for multi-dimensional analysis (Geography × Balance × Churn)

**Step 3: Interaction Detection**
- Analyzed feature combinations to uncover non-linear patterns (e.g., Age × Geography, NumOfProducts × Balance)
- Identified threshold effects where behavior changed dramatically (e.g., 2 vs 3 products)

**Step 4: Statistical Validation**
- Compared means, medians, and distributions between churners and stayers
- Dropped features with statistically insignificant differences (CreditScore, EstimatedSalary, HasCrCard, Tenure)

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

> Full visualizations and code in `notebook/bank_churn.ipynb`

---

## ⚖️ How F1 Score Was Improved: The Journey

### The Baseline Problem
80% stayed, 20% churned. Initial models trained on raw imbalanced data had a **cautious bias** — they optimized for accuracy by predicting "stay" most of the time.

**Baseline performance (without SMOTE):**
- Precision: 0.61 (when they predicted churn, they were right 61% of the time)
- Recall: 0.42 (they only caught 42% of actual churners)
- F1 Score: **0.50** (poor balance — missing most at-risk customers)

This is dangerous for business: the model was too conservative, letting 58% of churners slip through undetected.

### The SMOTE Strategy
SMOTE (Synthetic Minority Oversampling Technique) was applied to address class imbalance. Unlike simple duplication, SMOTE:
- **Interpolates** between existing minority samples in feature space
- Creates **synthetic, plausible examples** that fill the decision boundary
- Helps the model learn the true shape of the minority class

**Critical implementation detail:** SMOTE was applied **inside the pipeline on training data only** — no leakage into validation or test sets, preventing overoptimistic metrics.

### The Improvement
| Metric | Without SMOTE | With SMOTE | Change |
|---|---|---|---|
| Precision | 0.61 | 0.56 | -8% (acceptable trade-off) |
| Recall | 0.42 | 0.70 | **+67%** (major gain) |
| F1 Score | 0.50 | 0.62 | **+24%** |
| ROC-AUC | 0.82 | 0.87 | +6% |

**Why this trade-off works for churn prediction:**
- **False positive cost:** A retention email or phone call (~$5-10)
- **False negative cost:** Losing a customer worth thousands in lifetime value
- The model now catches 70% of churners instead of 42% — a **67% improvement in detection rate**

### Additional F1 Boosters
Beyond SMOTE, F1 was improved through:
1. **Feature selection** — dropping 4 noisy features (CreditScore, EstimatedSalary, HasCrCard, Tenure) that added variance without signal
2. **Gradient Boosting** — chosen over linear models for its ability to capture non-linear interactions (e.g., the 2-product sweet spot)
3. **Proper evaluation** — using stratified train/test split to preserve class ratios

---

## 🤖 Model Comparison

9 classifiers were benchmarked using the same SMOTE-augmented pipeline. Gradient Boosting emerged as the winner.

| Model | F1 Score | Accuracy | ROC-AUC |
|---|---|---|---|
| **Gradient Boosting** ✅ | **0.62** | 0.82 | **0.87** |
| Random Forest | 0.60 | 0.83 | 0.85 |
| AdaBoost | 0.58 | 0.79 | - |
| XGBoost | 0.57 | 0.77 | - |
| Extra Trees | 0.57 | 0.82 | - |
| Ridge Classifier | 0.50 | 0.71 | - |
| Logistic Regression | 0.50 | 0.71 | 0.76 |
| SGD Classifier | 0.48 | 0.69 | - |
| Decision Tree | 0.48 | 0.77 | 0.72 |

**Why Gradient Boosting won:**
- Churn patterns are **non-linear** (e.g., the 2-product sweet spot, Germany's double churn risk)
- Tree-based models capture interactions like `Age × Geography × NumOfProducts` that linear models miss
- Sequential error correction — each tree fixes the mistakes of the previous one
- The 0.12 F1 gap between Logistic Regression (0.50) and Gradient Boosting (0.62) represents the value of modeling complexity

> Full comparison in `notebook/bank_churn.ipynb`

---

## 💰 Real-World Impact: Business Value

The patterns discovered and the model built translate directly into actionable business interventions. Here's how this work creates value:

### Quantified Impact Estimates

**Assumptions:**
- Average customer lifetime value (CLV): $5,000
- Retention campaign cost per customer: $10
- Bank customer base: 10,000 (matching dataset)

**Detection Improvement:**
- Before: Model caught 42% of churners (F1 = 0.50)
- After: Model catches 70% of churners (F1 = 0.62)
- **+67% improvement in churn detection**

**Financial Impact Calculation:**
```
Annual churners (20% rate): 2,000 customers
Previously detected: 840 churners (42%)
Now detected: 1,400 churners (70%)
Additional churners detected: 560

If 30% of detected churners can be retained:
- Additional retained customers: 168
- Value saved: 168 × $5,000 = $840,000/year
- Campaign cost: 560 × $10 = $5,600
- Net ROI: $834,400/year (149× return)
```

### Actionable Interventions by Segment

| High-Risk Segment | Churn Rate | Recommended Action | Expected Impact |
|---|---|---|---|
| **Germany customers** | 32.4% | Dedicated Germany retention team, localized offers | Reduce to ~20% (save ~$250k/year) |
| **4-product holders** | 100% | Immediate outreach, product simplification offer | Prevent 100% attrition (save ~$500k/year) |
| **Inactive members** | 26.9% | Re-engagement campaigns, app notifications | Reduce to ~15% (save ~$120k/year) |
| **Age 45-60** | High | Personalized relationship management | Reduce by 20% (save ~$180k/year) |

### The 2-Product Sweet Spot Insight
The discovery that 2-product customers are the most loyal (7.6% churn) while 3+ product customers churn at 82%+ is a **strategic goldmine**:
- **Stop aggressive cross-selling** beyond 2 products
- **Simplify portfolios** for 3-4 product customers
- **Potential revenue protection:** Retaining high-balance customers who would otherwise leave due to product overload

### Deployment Value
The FastAPI application enables:
- **Real-time risk scoring** for customer service representatives
- **Automated alerts** when high-value customers enter risk zones
- **A/B testing** of retention strategies on predicted high-risk segments
- **Continuous monitoring** of model performance and drift

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

