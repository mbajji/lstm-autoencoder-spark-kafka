"""
Step 2c (fraud): Grid sweep -> retrain the winning configuration.

Explores a small grid of hyperparameters (hidden_dim, sequence_length,
learning_rate), ranks them by PR-AUC on the test split (the right headline
metric at 0.17% fraud prevalence), then RETRAINS the winning configuration
end-to-end and saves it to `models/credit_card/best/`.

Mirrors the taxi grid sweep (`code/4_grid_sweep.py`): the baseline lives in
models/credit_card/initial/, the sweep tells you what it tried, and the
retrained winner lands in models/credit_card/best/ for `2_evaluate_fraud.py`.

Usage:
    python code/4_grid_sweep_fraud.py
    python code/4_grid_sweep_fraud.py --epochs 30
"""

import argparse
import itertools
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_model
from src.scorer import AnomalyScorer, ScorerConfig
from src.preprocess_fraud import FraudPreprocessorConfig, preprocess_pipeline
from src.training import TrainingConfig, save_training_artifacts, train_model
from src.fraud_eval import make_loader, evaluate, print_metrics

logger = logging.getLogger(__name__)

# Search space (kept small so the sweep finishes in minutes on CPU).
HIDDEN_DIMS = [32, 64]
SEQ_LENS = [20, 30]
LEARNING_RATES = [1e-3, 5e-4]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(data_path, hidden_dim, seq_len, lr, epochs, batch_size, patience, device, seed):
    """Train a single configuration and return (model, scaler, scorer, history, pp_config, metrics)."""
    _set_seed(seed)
    pp_config = FraudPreprocessorConfig(sequence_length=seq_len, stride=seq_len)
    window_splits, label_splits, scaler, pp_config = preprocess_pipeline(data_path, pp_config)

    n_features = window_splits["train"].shape[2]
    train_loader = make_loader(window_splits["train"], batch_size, shuffle=True)
    val_loader = make_loader(window_splits["val"], batch_size, shuffle=False)

    model = create_model(input_dim=n_features, hidden_dim=hidden_dim,
                         num_layers=1, dropout=0.2, sequence_length=seq_len)
    model.to(device)

    train_cfg = TrainingConfig(epochs=epochs, learning_rate=lr, patience=patience)
    model, history = train_model(model, train_loader, val_loader, device, config=train_cfg)

    scorer = AnomalyScorer(ScorerConfig(scoring_mode="point", threshold_method="percentile"))
    scorer.fit(model, train_loader, device)

    metrics = evaluate(model, scorer, window_splits, label_splits, device, batch_size)
    scorer.point_threshold = metrics["threshold"]
    return model, scaler, scorer, history, pp_config, metrics


def main():
    parser = argparse.ArgumentParser(description="Grid sweep for fraud LSTM-AE")
    parser.add_argument("--data-path", type=str, default=str(PROJECT_ROOT / "data" / "creditcard.csv"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "models" / "credit_card" / "best"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = list(itertools.product(HIDDEN_DIMS, SEQ_LENS, LEARNING_RATES))

    print("\n" + "=" * 60)
    print("FRAUD GRID SWEEP (step 2c)")
    print("=" * 60)
    print(f"Baseline lives in: models/credit_card/initial/")
    print(f"Configurations to test: {len(grid)}  (ranked by PR-AUC)")
    print(f"Device: {device}")
    print("=" * 60)

    results = []
    for i, (hidden_dim, seq_len, lr) in enumerate(grid, 1):
        tag = f"hidden={hidden_dim} seq={seq_len} lr={lr:g}"
        print(f"\n[{i}/{len(grid)}] Training {tag} ...")
        _, _, _, _, _, metrics = train_one(
            args.data_path, hidden_dim, seq_len, lr,
            args.epochs, args.batch_size, args.patience, device, args.seed,
        )
        print(f"    PR-AUC={metrics['pr_auc']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  "
              f"F1={metrics['f1']:.2%}")
        results.append({"hidden_dim": hidden_dim, "seq_len": seq_len, "lr": lr, **metrics})

    results.sort(key=lambda r: r["pr_auc"], reverse=True)

    print("\n" + "=" * 60)
    print("SWEEP RANKING (by PR-AUC)")
    print("=" * 60)
    print(f"{'rank':>4}  {'hidden':>6}  {'seq':>4}  {'lr':>7}  {'PR-AUC':>7}  {'ROC-AUC':>7}  {'F1':>6}")
    for rank, r in enumerate(results, 1):
        print(f"{rank:>4}  {r['hidden_dim']:>6}  {r['seq_len']:>4}  {r['lr']:>7g}  "
              f"{r['pr_auc']:>7.4f}  {r['roc_auc']:>7.4f}  {r['f1']:>6.2%}")

    best = results[0]
    print(f"\nWinner: hidden={best['hidden_dim']} seq={best['seq_len']} lr={best['lr']:g} "
          f"(PR-AUC={best['pr_auc']:.4f})")
    print("Retraining winner end-to-end and saving to models/credit_card/best/ ...")

    model, scaler, scorer, history, pp_config, metrics = train_one(
        args.data_path, best["hidden_dim"], best["seq_len"], best["lr"],
        args.epochs, args.batch_size, args.patience, device, args.seed,
    )
    print_metrics(metrics, title="BEST-CONFIG TEST RESULTS (per transaction)")

    save_training_artifacts(
        output_dir=args.output_dir,
        model=model,
        scaler=scaler,
        scorer=scorer,
        history=history,
        preprocess_config=pp_config,
    )
    print(f"\nArtifacts saved to: {args.output_dir}/")
    print("Evaluate with: python code/2_evaluate_fraud.py --model-dir models/credit_card/best")
    print("=" * 60)


if __name__ == "__main__":
    main()
