# LSTM Autoencoder Anomaly Detection — One Architecture, Three Streaming Domains

A reproduction of the EncDec-AD model (Malhotra et al., 2016) — an LSTM
encoder-decoder that detects anomalies by **reconstruction error** — applied
to three very different real-time data streams a Fortune 500 company actually
deals with. The model trains on *normal* data only and flags anything it
reconstructs poorly as anomalous (no anomaly labels needed at training time).

The same core (`src/model.py`, `src/training.py`, `src/scorer.py`) drives all
three pipelines; only the data shape and preprocessing change.

| Domain | Data | Shape | Scripts |
|--------|------|-------|---------|
| **NYC taxi demand** | ~10k half-hourly counts | univariate, weekly seasonality | `taxi/` |
| **Credit-card fraud** | 284k transactions (Kaggle) | 29 features, 0.17% fraud | `fraud/` |
| **Network intrusion** | 2.5M flows (CICIDS2017) | 52 features, 17% attacks | `cicids/` |

Each domain follows the **same four-command journey**: train a baseline →
evaluate → grid sweep → evaluate the best config. This one README covers all
three; each folder also has its own README with extra detail.

---

## 1. Install

```bash
git clone <repo-url>
cd lstm-autoencoder-spark-kafka
uv sync
```

All commands below use `uv run python …`. You can equivalently call the
project venv directly with `.venv/bin/python …`.

## 2. Get the data

| Dataset | File in `data/` | Where to download | Committed? |
|---------|-----------------|-------------------|------------|
| NYC taxi | `nyc_taxi.csv` | included in repo | yes (small) |
| Credit-card fraud | `creditcard.csv` | <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> | no (~150 MB, gitignored) |
| CICIDS2017 | `cicids2017_cleaned.csv` | <https://www.unb.ca/cic/datasets/ids-2017.html>, <https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed> | no (~717 MB, gitignored) |

The two large CSVs exceed GitHub's 100 MB limit, so download them locally
into `data/`.

---

## 3. Run a pipeline

Pick a domain. Every domain has the same 4 steps (2a–2d).

### 3.1 NYC taxi demand  (`taxi/`)

```bash
uv run python taxi/1_train_model.py                                  # 2a baseline  -> models/initial/
uv run python taxi/2_evaluate_model.py                               # 2b evaluate + plots
uv run python taxi/4_grid_sweep.py                                   # 2c sweep     -> models/best/
uv run python taxi/2_evaluate_model.py --model-dir models/best       # 2d evaluate best
```

Detects holiday/event surges in taxi demand. The grid sweep takes the
baseline from ~83% F1 to 100% F1 (5/5 known events, zero false positives).
The Docker streaming demo (Kafka → Spark → Dash) uses `taxi/3_streaming_app.py`
— see `taxi/README.md`.

### 3.2 Credit-card fraud  (`fraud/`)

```bash
uv run python fraud/1_train_fraud.py                                         # 2a -> models/credit_card/initial/
uv run python fraud/2_evaluate_fraud.py                                      # 2b evaluate + plots
uv run python fraud/4_grid_sweep_fraud.py                                    # 2c sweep -> models/credit_card/best/
uv run python fraud/2_evaluate_fraud.py --model-dir models/credit_card/best  # 2d
```

Trains on legitimate transaction windows; flags fraud as high reconstruction
error. Fast smoke run: `fraud/1_train_fraud.py --max-train-windows 2000 --epochs 5`.

### 3.3 Network intrusion — CICIDS2017  (`cicids/`)

```bash
uv run python cicids/1_train_cicids.py                                   # 2a -> models/cicids/initial/
uv run python cicids/2_evaluate_cicids.py                                # 2b evaluate + plots
uv run python cicids/4_grid_sweep_cicids.py                              # 2c sweep -> models/cicids/best/
uv run python cicids/2_evaluate_cicids.py --model-dir models/cicids/best # 2d
```

Trains on benign flow windows; flags attacks (DoS/DDoS/PortScan/...) as high
reconstruction error. Fast smoke run: `cicids/1_train_cicids.py --max-rows 300000 --epochs 5`.

---

## 4. How to read the results

For the **fraud** and **CICIDS** pipelines the positive class is rare, so
accuracy is meaningless (predicting "all normal" already scores >99% on
fraud). The headline metrics are **PR-AUC** and **ROC-AUC** (threshold-free);
precision / recall / F1 are reported at the best-F1 threshold chosen on the
validation split — the test labels never set the threshold.

Each `2_evaluate_*` script also writes diagnostic plots to `evaluation/`
(score distributions + precision-recall curve).

## 5. Methodology (shared)

```
[----------- train (normal only) -----------][--- val ---][--- test ---]
```

1. **Preprocess** — scale features on normal training data only, slice the
   stream into fixed-length windows, keep only normal windows for training.
2. **Model** (`src/model.py`) — LSTM encoder-decoder reconstructs each window
   in reverse order (teacher forcing).
3. **Score** (`src/scorer.py`) — fit a Gaussian on normal reconstruction
   errors; anomaly score = Mahalanobis distance of the error.
4. **Threshold** — pick the best-F1 cut on validation, apply to test.

See `TECHNICAL.md` for architecture and scoring details.

## 6. Repository layout

```
src/         shared: model, training loop, scorer, preprocess_*, fraud_eval
taxi/        NYC taxi step scripts + README  (the original demo, incl. Docker/Striim)
fraud/       credit-card fraud step scripts + README
cicids/      CICIDS2017 intrusion step scripts + README
data/        datasets (large CSVs gitignored — see section 2)
models/      trained artifacts (gitignored)
evaluation/  diagnostic plots
striim/      Striim deployment for the streaming demo
TECHNICAL.md architecture + scoring reference
STRIIM.md    Striim setup guide
```
