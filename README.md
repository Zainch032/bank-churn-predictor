# Bank Customer Churn Predictor

End-to-end ML project — EDA, feature engineering, SMOTE, multi-model benchmarking, XGBoost tuning, and cost-based threshold selection.

**Live demo:** [huggingface.co/spaces/Zainch032/Bank_Churn](https://huggingface.co/spaces/Zainch032/Bank_Churn)

---

## The Problem

Customer churn is one of the most expensive problems in retail banking. When a customer closes their account or stops using services, the bank loses not just their current balance — it loses years of future transactions, loan interest, credit card fees, and cross-sell revenue. Studies consistently show that acquiring a new banking customer costs **5–7× more** than retaining an existing one.

What makes churn particularly hard to manage is that it's silent. A customer doesn't send a resignation letter or file a complaint before leaving. They simply stop using the app, let their balance drop to zero, and eventually move to a competitor. By the time the bank notices the pattern, the customer is already gone.

Traditional approaches — like reviewing accounts after they've been dormant for 90 days — are reactive and too late. The goal of this project is to shift that from reactive to **proactive**: identify customers who are at risk of churning *before* they decide to leave, so the retention team can reach out while there's still a window to act.

The dataset contains **10,000 bank customers** with 14 features covering demographics (age, gender, geography), financial behavior (balance, number of products, estimated salary), and account status (active member, credit card holder, tenure). The target variable is binary: did the customer churn within the observed period?

With a **20.4% churn rate**, the dataset is moderately imbalanced — a model that predicts "stay" for every customer hits 80% accuracy while catching zero actual churners. That's the baseline this project had to beat, and accuracy was never the metric that mattered.

---

## What I Built

| Phase | Detail |
|---|---|
| EDA | Crosstabs, boxenplots, pivot tables — every feature tested against the target |
| Feature Engineering | 5 interaction terms built from EDA findings |
| Resampling | SMOTETomek inside a leak-proof pipeline |
| Model Comparison | 9 classifiers benchmarked on F1, not accuracy |
| Tuning | XGBoost + RandomizedSearchCV (40 iterations, 5-fold CV, scoring=F1) |
| Threshold Selection | Two modes — F1-optimal and cost-optimal |
| Deployment | FastAPI REST API + frontend prediction form |

---

## Key EDA Findings

**Age — strongest signal**
Churners averaged 44.8 years vs 37.4 for stayers. The gap held across all geographies and genders. Ranked as top feature by XGBoost after training.

**Number of products — most non-linear pattern**

| Products | Churn Rate |
|---|---|
| 1 | 27.7% |
| 2 | 7.6% ← sweet spot |
| 3 | 82.7% |
| 4 | 100% |

Two products is the loyalty sweet spot. Three or more suggests over-selling. No linear model captures this — it's exactly why tree-based models outperform logistic regression here.

**Germany — double the churn risk**

| Country | Churn Rate |
|---|---|
| France | ~16% |
| Spain | ~16% |
| Germany | 32.4% |

German customers also hold the highest average balances (~$120k vs ~$62k for France). The most valuable customers by balance are also the most likely to leave.

**Active membership — most actionable lever**
Inactive members churn at 26.9% vs 14.3% for active members. Unlike age or geography, engagement can be directly changed through campaigns and notifications.

**Features dropped (with evidence)**

| Feature | Stayed Avg | Churned Avg | Decision |
|---|---|---|---|
| CreditScore | 651 | 645 | Dropped |
| EstimatedSalary | $100k | $101k | Dropped |
| HasCrCard | 70.5% | 70.9% | Dropped |
| Tenure | 5.03 yrs | 4.93 yrs | Dropped |

---

## Feature Engineering

Built 5 interaction terms directly from EDA findings:

| Feature | Rationale |
|---|---|
| `Zero_Balance` | Disengaged accounts carry $0 balance |
| `Balance_Salary_Ratio` | Financial sophistication independent of raw balance |
| `Products_Bucket` | Directly encodes the 2-vs-3+ product cliff (7.6% → 82.7%) |
| `Age_Inactive` | Interaction of the two strongest signals |
| `Germany_HighBal` | Isolates the specific high-risk segment from EDA |

---

## Model Results

9 classifiers benchmarked with SMOTE + same preprocessing pipeline. Final model: **XGBoost** tuned with RandomizedSearchCV (40 iterations, F1 scoring, 5-fold stratified CV).

### Threshold Selection

Threshold selection is a **business decision**, not just a number to maximize.

| Mode | Threshold | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| F1-optimal | ~0.545 | ~64% | ~63% | ~0.636 | ~85% |
| Cost-optimal | ~0.410 | ~75% | ~52% | ~0.615 | ~81% |

**Cost-optimal logic:** missing a churner costs ~$5,000 in lifetime value. A false positive costs $10 (a retention email). Under a 30% campaign contact-capacity constraint, the cost-optimal threshold minimizes total expected business loss — naturally pushing recall higher, but stopping at the point where more contacts stop creating net value.

**Why two modes matter:** a general deployment uses F1-optimal (balanced). When the retention team is running a campaign and has budget to contact more customers, cost-optimal mode is activated — it catches 75% of churners at the cost of more false alarms, which is acceptable when a retention email costs less than 1% of a customer's lifetime value.

---

## Business Impact

Based on the cost-optimal model at 10,000 customers (20% annual churn):

- Additional churners detected vs baseline: ~+50 per cycle
- If 30% of contacted churners are retained: ~15 additional retained customers
- At $5,000 CLV: ~$75,000 saved per cycle
- Campaign cost: negligible at $10/contact

**Actionable interventions by segment:**

| Segment | Churn Rate | Recommended Action |
|---|---|---|
| Germany customers | 32.4% | Dedicated retention team, localized offers |
| 3-4 product holders | 82-100% | Cap cross-sell at 2 products, simplify portfolios |
| Inactive members | 26.9% | Re-engagement campaigns, app notifications |
| Age 45-60 + high balance | High | Relationship manager assignment |

---

## Quick Start

```bash
git clone https://github.com/Zainch032/Bank_Churn.git
cd Bank_Churn
pip install -r requirements.txt
uvicorn main:app --reload
# UI  → http://127.0.0.1:8000
# API → http://127.0.0.1:8000/docs
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Data & ML | pandas · numpy · scikit-learn · imbalanced-learn · XGBoost |
| Visualization | matplotlib · seaborn |
| Backend | FastAPI · Uvicorn · Pydantic |
| Frontend | HTML · CSS · Vanilla JS |

---

## Author

**Muhammad Zain** — Final year AI Engineering student, Lahore, Pakistan

- GitHub: [Zainch032](https://github.com/Zainch032)
- LinkedIn: [Muhammad Zain](https://linkedin.com/in/muhammad-zain-9710692b4)
- Hugging Face: [Zainch032](https://huggingface.co/Zainch032)

---

MIT License
