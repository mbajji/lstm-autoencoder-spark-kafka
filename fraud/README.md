# LSTM Autoencoder Fraud Detection (Credit-Card Stream)

Forked from the NYC-taxi anomaly-detection pipeline and retargeted to
credit-card fraud — the Kaggle `creditcard.csv` benchmark (284,807
transactions, 492 frauds = 0.172%). Same EncDec-AD architecture
(Malhotra et al. 2016), now **multivariate**: each transaction is a 29-D
vector (V1–V28 + Amount). The model trains on *legitimate* windows only and
flags fraud as high reconstruction error (point-level Mahalanobis distance).

## Data

Download `creditcard.csv` from
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> and place it in `data/`.

## 1. Install dependencies

```bash
uv sync
```

## 2. Train a baseline, then improve it via grid sweep

A four-command journey. User-trained artifacts go to
`models/credit_card/initial/` and `models/credit_card/best/` (both gitignored).

#### 2a. Train the baseline

```bash
uv run python fraud/1_train_fraud.py
```

Trains a fast initial model (`hidden_dim=32`, `seq_len=30`, `10` epochs) on
legitimate transaction windows and writes artifacts to
`models/credit_card/initial/`.

#### 2b. Evaluate the baseline

```bash
uv run python fraud/2_evaluate_fraud.py
```

Reprints the per-transaction metrics and writes diagnostic plots to
`evaluation/`: `fraud_score_distribution.png` (legit vs fraud score
histograms) and `fraud_pr_curve.png` (precision-recall curve).

#### 2c. Run the grid sweep to find a better configuration

```bash
uv run python fraud/4_grid_sweep_fraud.py
```

Sweeps `hidden_dim × sequence_length × learning_rate`, ranks the
configurations by **PR-AUC**, then **retrains the winner end-to-end** and
saves it to `models/credit_card/best/`.

#### 2d. Evaluate the best-config model

```bash
uv run python fraud/2_evaluate_fraud.py --model-dir models/credit_card/best
```

Same evaluator, pointed at the retrained best artifacts.

## Why PR-AUC, not accuracy

At 0.17% fraud prevalence, predicting "all legit" already scores 99.8%
accuracy — so accuracy is meaningless here. The headline metrics are
**PR-AUC** and **ROC-AUC** (threshold-free). Precision / recall / F1 are
reported at the best-F1 threshold chosen on the validation split (the test
labels are never used to set the threshold).

## Methodology

```
[----------- train (legit only) -----------][--- val ---][--- test ---]
```

1. **Preprocess** (`src/preprocess_fraud.py`) — sort by `Time`, split
   chronologically to avoid leakage, scale the 29 features on legitimate
   training transactions only, slide a fixed-length window (30 transactions)
   over the stream, and drop any training window containing fraud.
2. **Model** (`src/model.py`) — LSTM encoder-decoder, `input_dim=29`,
   reconstructs each window in reverse order (teacher forcing).
3. **Score** (`src/scorer.py`) — fit a multivariate Gaussian on legit
   reconstruction errors; each transaction's anomaly score is its point-level
   Mahalanobis distance.
4. **Threshold** — choose the best-F1 cut on validation, apply to test.

## Project structure (fraud-specific files)

| Path | Role |
|------|------|
| `src/preprocess_fraud.py` | Load / scale / window the credit-card stream |
| `src/fraud_eval.py` | Shared per-transaction scoring + metrics |
| `fraud/1_train_fraud.py` | Step 2a — train baseline |
| `fraud/2_evaluate_fraud.py` | Steps 2b / 2d — evaluate + plots |
| `fraud/4_grid_sweep_fraud.py` | Step 2c — sweep + retrain winner |

Reused unchanged from the taxi pipeline: `src/model.py`, `src/training.py`,
and the multivariate path in `src/scorer.py`.
```
