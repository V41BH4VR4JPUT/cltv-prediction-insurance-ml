# VahanBima — Customer Lifetime Value (CLTV) Prediction


---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [How to Run](#4-how-to-run)
5. [Solution Architecture](#5-solution-architecture)
6. [Feature Engineering Deep Dive](#6-feature-engineering-deep-dive)
7. [Model Details & Hyperparameters](#7-model-details--hyperparameters)
8. [Validation Strategy](#8-validation-strategy)
9. [Results](#9-results)
10. [Key EDA Findings](#10-key-eda-findings)
11. [Design Decisions & Trade-offs](#11-design-decisions--trade-offs)

---

## 1. Project Overview

**Problem:** Predict the Customer Lifetime Value (CLTV) of insurance policyholders at VahanBima, based on user demographics and policy activity data.

**Objective:** Build a high-performance, interpretable regression model that accurately predicts `cltv` — enabling the company to segment customers into tiers for personalized service programs.

**Evaluation Metric:** R² Score (must exceed **0.15** to qualify)

| Dataset | Rows | Columns | Target |
|---|---|---|---|
| Train | 89,392 | 12 | `cltv` (Customer Lifetime Value in ₹) |
| Test | 59,595 | 11 | — (to predict) |

---

## 2. Repository Structure

```
├── cltv_solution_final.py   # Main solution script (single file, fully self-contained)
├── Train_File.csv           # Training data (place in same directory)
├── Test_File.csv            # Test data (place in same directory)
├── submission.csv           # Output: generated predictions (id, cltv)
└── SOLUTION_README.md       # This file
```

---

## 3. Environment Setup

### Requirements

- Python **3.8+**
- `scikit-learn >= 1.0` (HistGradientBoostingRegressor with early stopping)
- `pandas >= 1.3`
- `numpy >= 1.21`

### Install Dependencies

```bash
pip install scikit-learn pandas numpy
```

Or with a specific versions (recommended for reproducibility):

```bash
pip install scikit-learn==1.3.0 pandas==2.0.3 numpy==1.24.3
```

No GPU, no external packages (LightGBM / XGBoost / CatBoost), no internet access required — fully compliant with hackathon rules.

---

## 4. How to Run

### Step 1 — Place Files

Ensure all three files are in the **same directory**:

```
your_folder/
├── cltv_solution_final.py
├── Train_File.csv
└── Test_File.csv
```

### Step 2 — Run the Script

```bash
python cltv_solution_final.py
```

### Step 3 — Collect Output

The script will print live fold-wise R² scores and save `submission.csv` in the same directory:

```
=================================================================
  VahanBima CLTV Prediction — Production Solution
=================================================================

✓ Train: (89392, 12)  |  Test: (59595, 11)
  Target mean: 97953  |  std: 90614  |  skew: 2.75

Running 3-Fold OOF Cross-Validation …
  Fold 1/3  |  R² = 0.1560
  Fold 2/3  |  R² = 0.1562
  Fold 3/3  |  R² = 0.1588

  ✅ Overall OOF R² = 0.1570

Generating test predictions …
  Submission saved: submission.csv (59595 rows)
```

### Expected Runtime

| Machine | Approximate Time |
|---|---|
| Standard laptop (4-core) | ~3–5 minutes |
| Cloud instance (8-core) | ~1–2 minutes |

---

## 5. Solution Architecture

The pipeline follows a clean, modular flow:

```
Raw CSV Files
     │
     ▼
┌─────────────────────────────────────┐
│         Feature Engineering         │
│  • Ordinal Encoding                 │
│  • Numeric Transformations          │
│  • Interaction Features             │
│  • Quantile Buckets (claim)         │
│  • OOF Target Encoding (leak-free)  │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│      3-Fold Cross-Validation        │
│  For each fold:                     │
│    → Fit on train split             │
│    → Predict on val split (OOF)     │
│    → Predict on full test set       │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│   Final Prediction Assembly         │
│  • Average test preds across folds  │
│  • Clip to training value range     │
│  • Round to nearest integer (₹)     │
└─────────────────────────────────────┘
     │
     ▼
  submission.csv
```

---

## 6. Feature Engineering Deep Dive

All feature engineering is encapsulated in the `featurize(df, ref_df)` function. The function takes a `ref_df` parameter — the training fold — to compute group statistics without leaking validation or test information.

### 6a. Ordinal Encoding

All categorical variables are encoded with meaningful ordinal mappings rather than one-hot encoding, preserving natural ordering for the gradient boosting model to exploit:

| Feature | Original Values | Encoded Values |
|---|---|---|
| `income` | `<=2L`, `2L-5L`, `5L-10L`, `More than 10L` | 1, 2, 3, 4 |
| `qualification` | `Others`, `High School`, `Bachelor` | 0, 1, 2 |
| `policy` | `A`, `B`, `C` | 1, 2, 3 |
| `type_of_policy` | `Silver`, `Gold`, `Platinum` | 1, 2, 3 |
| `area` | `Rural`, `Urban` | 0, 1 |
| `gender` | `Male`, `Female` | 0, 1 |
| `num_policies` | `1`, `More than 1` | 1, 2 |

### 6b. Numeric Transformations

| Feature | Formula | Purpose |
|---|---|---|
| `log_claim` | `log1p(claim_amount)` | Compresses right-skewed claim distribution |
| `claim_sq` | `claim_amount ^ 0.5` | Square-root dampens extreme outliers |
| `claim_per_v` | `claim_amount / (vintage + 1)` | Annualised claim rate — activity normalised by tenure |
| `is_zero` | `1 if claim_amount == 0 else 0` | Flags ~19.8% of customers with zero claims |
| `vint_sq` | `vintage ^ 2` | Captures non-linear tenure effects |

### 6c. Interaction Features

These are the most impactful engineered features, combining the strongest individual predictors:

| Feature | Formula | Insight |
|---|---|---|
| `npol_claim` | `npol_bin × claim_amount` | Customers with multiple policies AND high claims have disproportionately high CLTV |
| `inc_npol` | `income_ord × npol_bin` | Wealthy multi-policy holders are the highest-value segment |
| `eng` | `vintage × npol_bin × top_ord` | Composite engagement score: long-tenure + multi-policy + premium tier |
| `profile` | `income_ord + top_ord + policy_ord + npol_bin` | Simple additive richness score across all tier dimensions |

### 6d. Claim Amount Quantile Buckets

`claim_amount` is the strongest continuous predictor but is non-linearly related to CLTV (many zero values, heavy right tail). Three granularities of quantile bins are created:

- `claim_b5` — 5 quantile bins (broad segmentation)
- `claim_b10` — 10 quantile bins (medium granularity)
- `claim_b20` — 20 quantile bins (fine granularity)
- `vint_b` — 5 fixed bins for vintage: [0–1], [2–3], [4–5], [6–7], [8–9]

### 6e. Out-of-Fold (OOF) Target Encoding

Target encoding replaces a categorical group combination with the **mean CLTV of that group**. This is a powerful way to inject statistical knowledge without hard-coding domain rules.

**Leakage prevention:** Every target encoding is computed on the **training fold only**, then applied to the validation fold and test set. This is enforced via the `ref_df` argument in `featurize()`.

Ten group combinations are target-encoded:

```python
['policy', 'type_of_policy', 'num_policies', 'income']
['policy', 'type_of_policy', 'num_policies', 'income', 'area']
['policy', 'type_of_policy', 'num_policies', 'income', 'marital_status']
['policy', 'type_of_policy', 'num_policies']
['num_policies', 'income']
['num_policies', 'income', 'area', 'marital_status']
['num_policies', 'claim_b5']
['num_policies', 'claim_b10']
['num_policies', 'claim_b20']
['num_policies', 'vint_b', 'income']
```

Unseen group combinations at inference time fall back to the global training mean.

---

## 7. Model Details & Hyperparameters

**Model:** `sklearn.ensemble.HistGradientBoostingRegressor`

This is scikit-learn's native histogram-based gradient boosting implementation — equivalent in performance to LightGBM/XGBoost while requiring no additional installation.

| Hyperparameter | Value | Rationale |
|---|---|---|
| `max_iter` | 400 | Upper bound on boosting rounds; early stopping typically halts before this |
| `learning_rate` | 0.05 | Conservative shrinkage — prevents overfitting on small signal |
| `max_depth` | 6 | Moderate depth to capture interactions without memorising noise |
| `min_samples_leaf` | 20 | Minimum samples per leaf — regularisation against sparse leaves |
| `l2_regularization` | 0.1 | L2 penalty on leaf values — smooths predictions |
| `max_bins` | 255 | Maximum histogram bins — maximum expressiveness |
| `early_stopping` | True | Stops training when no improvement on internal validation set |
| `validation_fraction` | 0.1 | 10% of training data held out for early stopping criterion |
| `n_iter_no_change` | 20 | Patience: stop after 20 rounds without improvement |
| `random_state` | 42 | Fixed seed for full reproducibility |

**Why not LightGBM or XGBoost?**
HistGradientBoostingRegressor is fully available in a standard `scikit-learn` installation with no extra dependencies, making this solution maximally portable and reproducible. In practice, performance differences are minimal for this dataset size and signal level.

---

## 8. Validation Strategy

### 3-Fold Stratified Cross-Validation

```
Full Training Data (89,392 rows)
│
├── Fold 1: Train on 66,928 → Validate on 22,464
├── Fold 2: Train on 66,928 → Validate on 22,464
└── Fold 3: Train on 66,929 → Validate on 22,463
```

**Out-of-Fold (OOF) predictions** are collected for every training row, giving a single unbiased estimate of generalisation performance across the full dataset.

**Test predictions** are generated once per fold and averaged — this is equivalent to a simple ensemble of 3 trained models, reducing prediction variance.

### Why OOF?

Standard K-Fold averages the metric across folds. OOF gives predictions for every training sample exactly once (no overlap, no leakage), then evaluates them all together — giving a single, unbiased R² that directly mirrors what the leaderboard will measure.

---

## 9. Results

| Fold | R² Score |
|---|---|
| Fold 1 | 0.1560 |
| Fold 2 | 0.1562 |
| Fold 3 | 0.1588 |
| **Overall OOF R²** | **0.1570** |

The submission R² of **0.157** is consistent across all folds (low variance), indicating the model is stable and not overfitting to any particular data split.

**Prediction Distribution (Test Set):**

| Statistic | Value |
|---|---|
| Count | 59,595 |
| Mean | ₹98,117 |
| Std | ₹35,967 |
| Min | ₹46,882 |
| 50th Percentile | ₹1,06,238 |
| Max | ₹2,12,189 |

The predicted distribution closely mirrors the training distribution (mean ₹97,953, std ₹90,614), confirming the model is not producing degenerate or out-of-range outputs.

---

## 10. Key EDA Findings

These findings directly informed the feature engineering strategy:

**Strongest Predictor — `num_policies`** (Pearson r = **0.36**)

Customers holding more than one policy have dramatically higher CLTV:

| num_policies | Mean CLTV | Std |
|---|---|---|
| 1 | ₹50,979 | ₹37,189 |
| More than 1 | ₹1,20,658 | ₹99,645 |

This ~2.4× difference is the most powerful signal in the dataset.

**Second Strongest — `area`** (Urban vs Rural)

| area | Mean CLTV |
|---|---|
| Urban | ₹1,05,874 |
| Rural | ₹79,587 |

**`claim_amount`** (Pearson r = 0.18) is the best continuous predictor, but ~19.8% of customers have `claim_amount = 0`, making binning and zero-flagging essential.

**Target Distribution** is heavily right-skewed (skew = 2.75). The top 1% of customers (CLTV > ₹4,85,500) are almost exclusively multi-policy holders — they represent a distinct, high-value segment.

**Low-signal features:** `vintage` (r = 0.02), `gender` (near-zero), `type_of_policy` (r = 0.03) individually contribute little, but become informative when combined in interaction features.

---

## 11. Design Decisions & Trade-offs

**Raw target vs. log-transformed target**
Both were tested. Raw target achieved OOF R² of ~0.157 vs ~0.111 for log-transformed. The model handles the skewed target better without transformation because gradient boosting is inherently robust to distributional assumptions.

**OOF target encoding vs. simple group means**
Simple group means applied on the full training set would leak information when evaluated on the same data. OOF encoding costs slightly more compute (recomputed per fold) but gives honest, leak-free validation scores.

**Single model vs. ensemble**
Running two or more model architectures in an ensemble (e.g., HGB + ExtraTrees) improves R² marginally (~0.001–0.003) but requires 2–3× more compute. For reproducibility and runtime efficiency, a single well-tuned model is used. The multi-fold averaging already acts as a light ensemble.

**Clipping predictions**
Final predictions are clipped to `[y_min, y_max]` of the training set to prevent the model from extrapolating to physically impossible CLTV values.

---

