"""
Data Preprocessing for LSTM Encoder-Decoder Fraud Detection

Adapts the EncDec-AD pipeline (Malhotra et al., 2016) from univariate,
weekly-seasonal NYC taxi demand to MULTIVARIATE credit-card transaction
streams (the Kaggle "creditcard.csv" benchmark: 284,807 transactions,
492 frauds = 0.172%).

Key differences from the taxi pipeline (src/preprocess.py):
- Multivariate: each transaction is a 29-D vector (V1..V28 + Amount),
  not a single scalar. `input_dim = 29`.
- No weekly seasonality. Instead of 336-step weeks we slide a short,
  fixed-length window over the transaction stream (default 30 events).
- Labels are GROUND TRUTH (the `Class` column), not inferred from
  holiday date windows.
- Unsupervised setup: the model trains on LEGITIMATE windows only, so
  fraud shows up as high reconstruction error (out-of-distribution).

Splits are chronological (by `Time`) to avoid leakage:
    [-------- train (legit only) --------][--- val ---][--- test ---]
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Kaggle creditcard.csv schema
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]  # 29 features
LABEL_COLUMN = "Class"
TIME_COLUMN = "Time"


@dataclass
class FraudPreprocessorConfig:
    """Configuration for credit-card fraud preprocessing."""
    sequence_length: int = 30        # transactions per window
    stride: int = 30                 # non-overlapping windows (stride == length)
    train_fraction: float = 0.60     # chronological share used for training
    val_fraction: float = 0.20       # next share for validation / thresholding
    # remaining (~0.20) is the test set
    feature_columns: List[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and parse the Kaggle credit-card CSV.

    Returns a DataFrame sorted by Time with the expected columns present.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place creditcard.csv in the data/ directory."
        )

    df = pd.read_csv(filepath)

    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    if TIME_COLUMN in df.columns:
        df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    n_fraud = int(df[LABEL_COLUMN].sum())
    logger.info(f"Loaded {len(df):,} transactions from {filepath}")
    logger.info(
        f"  Fraud: {n_fraud:,} ({n_fraud / len(df):.3%})  "
        f"Legit: {len(df) - n_fraud:,}"
    )
    return df


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def make_windows(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_length: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a fixed-length window over a contiguous block of transactions.

    Args:
        features: (N, F) scaled feature matrix
        labels:   (N,)   per-transaction labels (0 legit / 1 fraud)
        sequence_length: window length L
        stride:   step between window starts

    Returns:
        windows:       (num_windows, L, F)
        window_labels: (num_windows, L)  -- per-position labels
    """
    n = len(features)
    if n < sequence_length:
        return (
            np.empty((0, sequence_length, features.shape[1]), dtype=np.float32),
            np.empty((0, sequence_length), dtype=np.int64),
        )

    starts = range(0, n - sequence_length + 1, stride)
    windows = np.stack([features[s:s + sequence_length] for s in starts]).astype(np.float32)
    window_labels = np.stack([labels[s:s + sequence_length] for s in starts]).astype(np.int64)
    return windows, window_labels


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_pipeline(
    filepath: str,
    config: Optional[FraudPreprocessorConfig] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], StandardScaler, FraudPreprocessorConfig]:
    """
    Run the complete fraud preprocessing pipeline.

    Steps:
        1. Load + sort by Time.
        2. Chronological split into train / val / test transaction blocks.
        3. Fit StandardScaler on LEGITIMATE training transactions only.
        4. Build non-overlapping windows per split.
        5. Drop training windows that contain any fraud (unsupervised setup).

    Returns:
        Tuple of (window_splits, label_splits, scaler, config) where
        window_splits has keys train/val/test -> (n, L, F) arrays and
        label_splits has the matching per-position labels (n, L).
    """
    config = config or FraudPreprocessorConfig()

    logger.info("=" * 60)
    logger.info("Starting fraud preprocessing pipeline")
    logger.info("=" * 60)

    df = load_data(filepath)
    features_all = df[config.feature_columns].to_numpy(dtype=np.float32)
    labels_all = df[LABEL_COLUMN].to_numpy(dtype=np.int64)

    n = len(df)
    train_end = int(n * config.train_fraction)
    val_end = int(n * (config.train_fraction + config.val_fraction))
    logger.info(
        f"Chronological split: train[0:{train_end}] "
        f"val[{train_end}:{val_end}] test[{val_end}:{n}]"
    )

    # Fit scaler on legitimate training transactions only.
    train_block_feats = features_all[:train_end]
    train_block_labels = labels_all[:train_end]
    legit_mask = train_block_labels == 0
    scaler = StandardScaler()
    scaler.fit(train_block_feats[legit_mask])
    logger.info(f"Fitted StandardScaler on {int(legit_mask.sum()):,} legit training transactions")

    features_scaled = scaler.transform(features_all).astype(np.float32)

    # Window each split block independently (windows never straddle a split).
    blocks = {
        "train": (features_scaled[:train_end], labels_all[:train_end]),
        "val": (features_scaled[train_end:val_end], labels_all[train_end:val_end]),
        "test": (features_scaled[val_end:], labels_all[val_end:]),
    }

    window_splits: Dict[str, np.ndarray] = {}
    label_splits: Dict[str, np.ndarray] = {}
    for name, (feats, labs) in blocks.items():
        windows, win_labels = make_windows(
            feats, labs, config.sequence_length, config.stride
        )
        if name == "train":
            # Unsupervised: keep only pure-legit windows for training.
            pure_legit = win_labels.sum(axis=1) == 0
            windows = windows[pure_legit]
            win_labels = win_labels[pure_legit]
            logger.info(
                f"  train: {int(pure_legit.sum()):,} pure-legit windows "
                f"(dropped {int((~pure_legit).sum()):,} windows containing fraud)"
            )
        else:
            n_fraud_windows = int((win_labels.sum(axis=1) > 0).sum())
            logger.info(
                f"  {name}: {len(windows):,} windows "
                f"({n_fraud_windows:,} contain >=1 fraud)"
            )
        window_splits[name] = windows
        label_splits[name] = win_labels

    logger.info("=" * 60)
    logger.info("Fraud preprocessing complete")
    logger.info("=" * 60)

    return window_splits, label_splits, scaler, config
