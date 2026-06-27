"""
Data Preprocessing for LSTM Encoder-Decoder Network-Intrusion Detection

Third domain for the EncDec-AD pipeline (after NYC-taxi demand and
credit-card fraud): network traffic from the CICIDS2017 benchmark
(Canadian Institute for Cybersecurity). The cleaned CSV holds ~2.52M
bidirectional flow records, 52 numeric flow features, and an `Attack Type`
label that is "Normal Traffic" for 83% of flows and one of six attack
classes (DoS, DDoS, Port Scanning, Brute Force, Web Attacks, Bots) for the
rest (~17%).

Same shape as the fraud problem (multivariate per-event records with ground
truth), so this mirrors `src/preprocess_fraud.py` and reuses the generic
evaluator in `src/fraud_eval.py`. Differences:
- 52 features instead of 29.
- Label is a STRING -> binarized as Normal Traffic (0) vs attack (1).
- No timestamp column. We window flows in file (capture) order: network
  attacks (DoS/DDoS/PortScan) arrive in BURSTS of consecutive malicious
  flows, so a window of consecutive flows is a meaningful sequence and the
  LSTM's sequence modelling is justified.
- CICIDS is notorious for non-finite values (Infinity in the per-second rate
  columns) and NaNs; we coerce to numeric, replace +/-inf with NaN, and fill
  with the per-column median of the legitimate training block.

Splits are positional (capture order) to avoid leakage:
    [-------- train (benign only) --------][--- val ---][--- test ---]
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

LABEL_COLUMN = "Attack Type"
NORMAL_LABEL = "Normal Traffic"


@dataclass
class CICIDSPreprocessorConfig:
    """Configuration for CICIDS2017 preprocessing."""
    sequence_length: int = 20        # flows per window
    stride: int = 20                 # non-overlapping windows
    train_fraction: float = 0.60
    val_fraction: float = 0.20
    max_rows: Optional[int] = None   # cap rows read (smoke runs); None = all
    feature_columns: Optional[List[str]] = field(default=None)  # default: all but label


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(filepath: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load and clean the CICIDS2017 cleaned CSV.

    Coerces feature columns to numeric, replaces +/-inf with NaN (filled
    later from training-block medians), and adds a binary `is_attack` column.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Download CICIDS2017 from https://www.unb.ca/cic/datasets/ids-2017.html "
            "(cleaned/merged CSV) and place it in the data/ directory."
        )

    df = pd.read_csv(filepath, nrows=max_rows, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"CSV is missing the label column '{LABEL_COLUMN}'")

    # Binary label: attack (1) vs normal (0).
    df["is_attack"] = (df[LABEL_COLUMN].astype(str).str.strip() != NORMAL_LABEL).astype(np.int64)

    n_attack = int(df["is_attack"].sum())
    logger.info(f"Loaded {len(df):,} flows from {filepath}")
    logger.info(
        f"  Attack: {n_attack:,} ({n_attack / len(df):.3%})  "
        f"Normal: {len(df) - n_attack:,}"
    )
    return df


def _resolve_features(df: pd.DataFrame, config: CICIDSPreprocessorConfig) -> List[str]:
    """Numeric feature columns = everything except the label / derived columns."""
    if config.feature_columns is not None:
        return list(config.feature_columns)
    drop = {LABEL_COLUMN, "is_attack"}
    return [c for c in df.columns if c not in drop]


# ---------------------------------------------------------------------------
# Windowing  (identical contract to preprocess_fraud.make_windows)
# ---------------------------------------------------------------------------

def make_windows(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_length: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a fixed-length window over a contiguous block of flows."""
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
    config: Optional[CICIDSPreprocessorConfig] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], StandardScaler, CICIDSPreprocessorConfig]:
    """
    Run the complete CICIDS preprocessing pipeline.

    Returns (window_splits, label_splits, scaler, config) with train/val/test
    window arrays (n, L, F) and matching per-position labels (n, L).
    """
    config = config or CICIDSPreprocessorConfig()

    logger.info("=" * 60)
    logger.info("Starting CICIDS2017 preprocessing pipeline")
    logger.info("=" * 60)

    df = load_data(filepath, max_rows=config.max_rows)
    feature_cols = _resolve_features(df, config)

    features_all = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float64)
    )
    labels_all = df["is_attack"].to_numpy(dtype=np.int64)

    n = len(df)
    train_end = int(n * config.train_fraction)
    val_end = int(n * (config.train_fraction + config.val_fraction))
    logger.info(
        f"Positional split: train[0:{train_end}] "
        f"val[{train_end}:{val_end}] test[{val_end}:{n}]"
    )

    # Impute NaNs (incl. former inf) with medians from the legit training block.
    train_block = features_all[:train_end]
    train_legit = train_block[labels_all[:train_end] == 0]
    medians = np.nanmedian(train_legit, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)  # all-NaN columns -> 0
    nan_idx = np.isnan(features_all)
    if nan_idx.any():
        features_all[nan_idx] = np.take(medians, np.where(nan_idx)[1])
        logger.info(f"Imputed {int(nan_idx.sum()):,} non-finite/NaN values with training medians")

    # Fit scaler on legitimate training flows only.
    scaler = StandardScaler()
    scaler.fit(features_all[:train_end][labels_all[:train_end] == 0])
    logger.info(
        f"Fitted StandardScaler on {int((labels_all[:train_end] == 0).sum()):,} "
        f"legit training flows ({len(feature_cols)} features)"
    )
    features_scaled = scaler.transform(features_all).astype(np.float32)

    blocks = {
        "train": (features_scaled[:train_end], labels_all[:train_end]),
        "val": (features_scaled[train_end:val_end], labels_all[train_end:val_end]),
        "test": (features_scaled[val_end:], labels_all[val_end:]),
    }

    window_splits: Dict[str, np.ndarray] = {}
    label_splits: Dict[str, np.ndarray] = {}
    for name, (feats, labs) in blocks.items():
        windows, win_labels = make_windows(feats, labs, config.sequence_length, config.stride)
        if name == "train":
            pure_benign = win_labels.sum(axis=1) == 0
            windows = windows[pure_benign]
            win_labels = win_labels[pure_benign]
            logger.info(
                f"  train: {int(pure_benign.sum()):,} pure-benign windows "
                f"(dropped {int((~pure_benign).sum()):,} windows containing attacks)"
            )
        else:
            n_attack_windows = int((win_labels.sum(axis=1) > 0).sum())
            logger.info(
                f"  {name}: {len(windows):,} windows ({n_attack_windows:,} contain >=1 attack)"
            )
        window_splits[name] = windows
        label_splits[name] = win_labels

    logger.info("=" * 60)
    logger.info("CICIDS2017 preprocessing complete")
    logger.info("=" * 60)

    return window_splits, label_splits, scaler, config
