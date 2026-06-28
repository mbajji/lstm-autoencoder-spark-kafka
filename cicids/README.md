# LSTM Autoencoder Network-Intrusion Detection (CICIDS2017)

Forked from the NYC-taxi anomaly-detection pipeline and retargeted to
**network-intrusion detection** — the CICIDS2017 benchmark from the Canadian
Institute for Cybersecurity (~2.52M bidirectional flow records, 52 numeric
flow features, ~17% attacks). Same EncDec-AD architecture (Malhotra et al.
2016), multivariate (`input_dim=52`): the model trains on *benign* traffic
only and flags attacks (DoS, DDoS, Port Scanning, Brute Force, Web Attacks,
Bots) as high reconstruction error (point-level Mahalanobis distance).

This is the third domain on one architecture, alongside the taxi demand
(`README.md`) and credit-card fraud (`README_FRAUD.md`) pipelines.

## Data

Download CICIDS2017 from
<https://www.unb.ca/cic/datasets/ids-2017.html> (the cleaned/merged CSV with
an `Attack Type` label) and place `cicids2017_cleaned.csv` in `data/`.
The file is ~717 MB, so it is gitignored — download it locally.

## 1. Install dependencies

```bash
uv sync
```

## 2. Train a baseline, then improve it via grid sweep

A four-command journey. User-trained artifacts go to `models/cicids/initial/`
and `models/cicids/best/` (both gitignored). The CICIDS step scripts live in
the `cicids/` folder.

#### 2a. Train the baseline

```bash
uv run python cicids/1_train_cicids.py
```

Trains a fast initial model (`hidden_dim=32`, `seq_len=20`, `10` epochs) on
benign flow windows and writes artifacts to `models/cicids/initial/`.

> Tip: add `--max-rows 300000 --epochs 5` for a fast smoke run on a slice of
> the 2.5M flows.

#### 2b. Evaluate the baseline

```bash
uv run python cicids/2_evaluate_cicids.py
```

Reprints the per-flow metrics and writes diagnostic plots to `evaluation/`:
`cicids_score_distribution.png` (benign vs attack score histograms) and
`cicids_pr_curve.png` (precision-recall curve).

#### 2c. Run the grid sweep to find a better configuration

```bash
uv run python cicids/4_grid_sweep_cicids.py
```

Sweeps `hidden_dim × sequence_length × learning_rate`, ranks the
configurations by **PR-AUC**, then **retrains the winner end-to-end** and
saves it to `models/cicids/best/`.

#### 2d. Evaluate the best-config model

```bash
uv run python cicids/2_evaluate_cicids.py --model-dir models/cicids/best
```

Same evaluator, pointed at the retrained best artifacts.

## Why PR-AUC, not accuracy

Attacks are the minority class and several types (DoS/DDoS/PortScan) are
volumetric and easy, while others (Brute Force/Web/Bots) are subtle — so
accuracy is misleading. The headline metrics are **PR-AUC** and **ROC-AUC**
(threshold-free); precision / recall / F1 are reported at the best-F1
threshold chosen on the validation split (test labels are never used to set
the threshold).

## Methodology

```
[----------- train (benign only) -----------][--- val ---][--- test ---]
```

1. **Preprocess** (`src/preprocess_cicids.py`) — binarize `Attack Type`
   (Normal Traffic = 0, any attack = 1); coerce features to numeric, replace
   `Infinity`/NaN with per-column medians from the benign training block;
   scale on benign training flows only; slide a fixed-length window
   (20 flows) over the stream; drop any training window containing an attack.
   Network attacks arrive in **bursts** of consecutive malicious flows, so a
   window of consecutive flows is a meaningful sequence.
2. **Model** (`src/model.py`) — LSTM encoder-decoder, `input_dim=52`,
   reconstructs each window in reverse order (teacher forcing).
3. **Score** (`src/scorer.py`) — fit a multivariate Gaussian on benign
   reconstruction errors; each flow's anomaly score is its point-level
   Mahalanobis distance.
4. **Threshold** — choose the best-F1 cut on validation, apply to test.

## Project structure (CICIDS-specific files)

| Path | Role |
|------|------|
| `src/preprocess_cicids.py` | Load / clean / scale / window CICIDS2017 flows |
| `src/fraud_eval.py` | Shared per-record scoring + metrics (used by fraud + CICIDS) |
| `1_train_cicids.py` | Step 2a — train baseline |
| `2_evaluate_cicids.py` | Steps 2b / 2d — evaluate + plots |
| `4_grid_sweep_cicids.py` | Step 2c — sweep + retrain winner |

Reused unchanged from the taxi pipeline: `src/model.py`, `src/training.py`,
and the multivariate path in `src/scorer.py`.
```
