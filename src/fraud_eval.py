"""
Shared evaluation helpers for the credit-card fraud pipeline.

Lives here so the baseline trainer (`code/1_train_fraud.py`), the evaluator
(`code/2_evaluate_fraud.py`), and the grid sweep (`code/4_grid_sweep_fraud.py`)
all score models the same way -- the fraud-domain analogue of how the taxi
scripts share `src/training.py`.

Scoring is per transaction: each window position gets a point-level
Mahalanobis score (Malhotra et al. 2016, multivariate path), windows are
flattened back to transactions, and we compare against the ground-truth
`Class` labels. Because fraud is ~0.17% of traffic, the headline metrics are
PR-AUC and ROC-AUC (threshold-free); precision/recall/F1 are reported at the
best-F1 threshold chosen on the validation split.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.scorer import AnomalyScorer

logger = logging.getLogger(__name__)


def make_loader(windows: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """DataLoader yielding (batch, seq_len, features) tensors."""
    tensor = torch.from_numpy(windows.astype(np.float32))
    ds = TensorDataset(tensor)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda b: torch.stack([x[0] for x in b]),
    )


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Scan candidate thresholds, return (threshold, f1) maximizing F1."""
    if labels.sum() == 0:
        # No fraud in this split -> fall back to a high percentile.
        return float(np.percentile(scores, 99.9)), 0.0
    candidates = np.unique(np.percentile(scores, np.linspace(90, 99.99, 200)))
    best_t, best_f1 = float(candidates[0]), -1.0
    for t in candidates:
        pred = scores > t
        _, _, f1, _ = precision_recall_fscore_support(
            labels, pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1


def evaluate(
    model: torch.nn.Module,
    scorer: AnomalyScorer,
    window_splits: Dict[str, np.ndarray],
    label_splits: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> Dict:
    """
    Score the val + test splits and compute per-transaction fraud metrics.

    The best-F1 threshold is chosen on validation and applied to test, so the
    reported precision/recall/F1 reflect a threshold picked without touching
    the test labels. Returns a metrics dict plus the raw test scores/labels
    (handy for plotting).
    """
    val_loader = make_loader(window_splits["val"], batch_size, shuffle=False)
    test_loader = make_loader(window_splits["test"], batch_size, shuffle=False)

    val_point_scores, _, _ = scorer.compute_point_scores(model, val_loader, device)
    test_point_scores, _, _ = scorer.compute_point_scores(model, test_loader, device)

    val_scores = val_point_scores.reshape(-1)
    val_labels = label_splits["val"].reshape(-1)
    test_scores = test_point_scores.reshape(-1)
    test_labels = label_splits["test"].reshape(-1)

    pr_auc = float(average_precision_score(test_labels, test_scores))
    roc_auc = float(roc_auc_score(test_labels, test_scores))

    threshold, _ = best_f1_threshold(val_scores, val_labels)
    test_pred = test_scores > threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(test_labels, test_pred, labels=[0, 1]).ravel()

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_transactions": int(len(test_labels)),
        "n_fraud": int(test_labels.sum()),
        "test_scores": test_scores,
        "test_labels": test_labels,
    }


def print_metrics(metrics: Dict, title: str = "TEST RESULTS (per transaction)") -> None:
    """Pretty-print an evaluate() result."""
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)
    print(f"  Records scored: {metrics['n_transactions']:,}  "
          f"(positives: {metrics['n_fraud']:,})")
    print(f"  PR-AUC  (avg precision): {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:                 {metrics['roc_auc']:.4f}")
    print(f"  Threshold (best-F1 @val): {metrics['threshold']:.4f}")
    print(f"  Precision: {metrics['precision']:.2%}   "
          f"Recall: {metrics['recall']:.2%}   F1: {metrics['f1']:.2%}")
    print(f"  Confusion: TP={metrics['tp']}  FP={metrics['fp']}  "
          f"FN={metrics['fn']}  TN={metrics['tn']}")
    print("-" * 60)
