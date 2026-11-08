# Bank Customer Churn: A Retention Decision

End-to-end churn project: business-first EDA, evidence-driven feature engineering, leak-proof resampling, 9-model benchmark, XGBoost tuning, and a **cost-based contact cutoff** that respects call-centre capacity.

**Live demo:** [huggingface.co/spaces/Zainch032/Bank_Churn](https://huggingface.co/spaces/Zainch032/Bank_Churn)

---

## The Business Problem

The book has **10,000 customers** and a **20.4% churn rate**: **2,037 customers left**, taking about **$186 million** in deposits. Churn here is quiet. There is no resignation letter; by the time an account has been dormant for months, the relationship is usually gone. Replacing a customer costs several times more than keeping one.

The goal is to move the bank from a late review to a **ranked list of who to contact while there is still something to save**.

A model is only useful if three questions are answered first:

1. **Who is expensive to lose?** (age, balance, Germany, product count)
2. **What can the bank actually change?** (activity and product policy, not age or country)
3. **How many people can the team contact?** Missing a churner is planned at about **$5,000** of lifetime value; a wasted contact at about **$10**. The cutoff is a budget decision, not the default 0.5.

> A model that predicts "stay" for everyone is 80% accurate and catches zero churners. Accuracy is the wrong scorecard, so this project uses F1, recall and cost.

**Dataset:** [Bank Customer Churn (Kaggle)](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction) · 10,000 customers · 14 fields · target `Exited`.

---

## Approach

| Step | Business reason |
|---|---|
| Data quality check (0 duplicates, 0 missing) | A retention list built on bad rows wastes call-centre budget |
| Drop `RowNumber`, `CustomerId`, `Surname` | Identifiers let a model memorise people instead of learning patterns |
| Test every attribute against churn | Drop anything that does not change who gets contacted |
| Engineer only what EDA proved | The product cliff and the Germany–balance pocket are not linear |
| Stratified 80/20 split, resample **inside** the training pipeline | The test set keeps the real 20% churn rate, so scores stay honest |
| Score on F1, then choose the cutoff from cost + a 30% capacity cap | Precision and recall translated into money and workload |

---

## Key EDA Findings

| Finding | Evidence | Decision |
|---|---|---|
| **Age is the strongest split** | Churners average 44.8 yrs vs 37.4. Over-41s churn at **39.9%** vs **11.1%** for 41 and under (~3.6×) | Use age to prioritise coverage, not as the campaign message |
| **Inactivity is the lever the bank can pull** | Inactive members churn at **26.9%** vs **14.3%**; ~64% of all exits are inactive customers | Fund re-activation first, ranked by balance |
| **Germany is a market problem** | **32.4%** churn vs ~16% in France/Spain; ~$120k avg balance vs ~$62k; ~40% of all exits | Local desk and offer rather than a global email |
| **Product count is non-linear** | 1 product: 27.7% · **2 products: 7.6%** · 3: 82.7% · 4: 100% | Cap cross-sell at 2; case-review the 60 four-product customers |
| **High balances leave** | Leavers hold more than stayers at every tenure year (~$84–98k vs ~$70–75k) | Put relationship-manager time on high-balance customers |
| **Zero-balance accounts are stable** | 13.8% churn vs 24.1% for funded accounts; holds within the young group (7.1% vs 13.5%) | Spend on customers with a balance |
| **Gender gap is real** | Women 25.1% vs men 16.5% | Diagnose it (service, product fit); do **not** price or target by gender |
| **Sharpest pocket** | Women over 41 with a balance: **51.8%** churn, 1,012 customers (10% of book), ~26% of all churners | Cover the whole over-41-with-balance list, applied evenly |

**Little or no signal:** credit score (652 vs 645 avg), estimated salary (churn 19.3–21.9% across deciles; Mann-Whitney p = 0.23), credit-card ownership (~20% either way) and tenure (churn roughly 17–23% by year). These are not used as targeting rules.

---

## Feature Engineering

Five features, each tied to a pattern EDA already showed:

| Feature | Why it exists |
|---|---|
| `Zero_Balance` | Empty accounts churn at 13.8%, funded at 24.1% |
| `Balance_Salary_Ratio` | Salary is flat; the ratio asks whether the deposit is large relative to income |
| `Products_Bucket` (3+ products) | Encodes the 7.6% → 83–100% cliff |
| `Age_Inactive` | The two strongest signals combined |
| `Germany_HighBal` | Germany with above-median balance, the market where rate and balance are both worst |

**Model inputs:** `Age`, `Balance`, `CreditScore`, `EstimatedSalary`, `NumOfProducts`, `IsActiveMember`, the five engineered features, plus `Geography` and `Gender`. `Tenure` and `HasCrCard` are excluded. Numeric fields are scaled and categoricals encoded inside a `ColumnTransformer`.

---

## Modelling

1. **Benchmark:** nine classifiers (Logistic Regression, Random Forest, Gradient Boosting, Decision Tree, AdaBoost, Extra Trees, Ridge, SGD, XGBoost) compared on F1 with SMOTE inside the pipeline. Class weights or `scale_pos_weight` are used where the model supports them.
2. **Tuning:** trees suit the product cliff and stacked risks, so XGBoost is tuned with `RandomizedSearchCV` (40 candidates, 5-fold stratified CV, scoring = F1) using **SMOTETomek**, which cleans noisy synthetic points before the model learns from them.
3. **Best CV F1:** 0.6239.
4. **Leakage control:** resampling runs only on training folds. The 2,000-customer test set (407 churners) is never resampled.

### Two operating modes (held-out test set)

| Mode | Threshold | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| **F1-optimal** (standing list) | 0.545 | 64.1% | 63.0% | 0.636 | ~85% |
| **Cost-optimal** (funded campaign) | 0.410 | 74.7% | 52.1% | 0.614 | ~81% |

**Cost-optimal logic:** a missed churner is planned at about $5,000 in lifetime value, a wasted contact at about $10, contacts are capped at 30% of the book, and a 30% save rate is assumed. The chosen cutoff contacts **29.2%** of the test set and catches ~75% of churners.

> The $5,000 / $10 / 30% figures are **planning assumptions**, not measured in the dataset. Re-run the threshold section with the bank's real values. The segment findings above do not depend on them.

---

## What I'd Take to the Retention Meeting

Of 2,037 exits and ~$186M in deposits:

| Priority | Action |
|---|---|
| Over 41, with a balance (~45% churn) | Named relationship coverage and a reviewed rate, applied evenly to all customers |
| Germany | Local desk and offers, with retained balance tracked separately |
| One product, high balance (1,409 exits, ~$130M) | Move to a **second** product; do not sell a third |
| Inactive members | Re-activation campaign, ranked by balance |
| 3–4 products | Policy: cap cross-sell at 2; call all 60 four-product customers rather than fund a campaign |
| Gender gap | Service review, not a gendered offer |

The model does not replace these segments. It **ranks inside them**, so a team that can contact only 30% of the book calls the highest expected-loss customers first. Measure each offer against a holdout group and stop it if it does not beat the holdout.

---

## Project Structure

```
Bank_Churn/
├── data/            # Churn_Modelling.csv, Cleaned_data.csv
├── notebooks/       # bank_churn.ipynb (EDA → model → export)
├── model/           # model_pipeline.pkl (full pipeline, not a bare classifier)
├── main.py          # FastAPI app
└── requirements.txt
```

The exported file is the **entire pipeline** (encoding, scaling, model), so the app can send raw, unscaled fields and get the same score the notebook would produce.

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
| Data & ML | pandas · numpy · scikit-learn · imbalanced-learn · XGBoost · SciPy |
| Visualization | matplotlib · seaborn |
| Backend | FastAPI · Uvicorn · Pydantic |
| Frontend | HTML · CSS · Vanilla JS |

---

## Author

**Muhammad Zain**: Final-year AI Engineering student, Lahore, Pakistan

- Portfolio: [zainai.me](https://zainai.me)
- GitHub: [Zainch032](https://github.com/Zainch032)
- Kaggle: [zain032](https://www.kaggle.com/zain032)
- LinkedIn: [Muhammad Zain](https://linkedin.com/in/muhammad-zain-9710692b4)
- Hugging Face: [Zainch032](https://huggingface.co/Zainch032)

---

MIT License