"""
Step 2a (fraud): Train the baseline.

Trains an LSTM Encoder-Decoder on the Kaggle credit-card stream and saves
artifacts to `models/credit_card/initial/`. This mirrors the taxi baseline
(`fraud/1_train_model.py`): a deliberately small, fast configuration
(hidden_dim=32, seq_len=30, 10 epochs) that learns the legitimate-transaction
manifold well enough to flag fraud as reconstruction error, but leaves clear
headroom for the grid sweep (`fraud/4_grid_sweep_fraud.py`) to improve on.

The model trains on LEGITIMATE windows only; fraud is detected as high
point-level Mahalanobis distance (multivariate path, Malhotra et al. 2016).

Usage:
    python fraud/1_train_fraud.py
    python fraud/1_train_fraud.py --epochs 30 --hidden-dim 64
    python fraud/1_train_fraud.py --output-dir models/credit_card/my_run
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Train baseline LSTM-AE for credit-card fraud")
    parser.add_argument("--data-path", type=str, default=str(PROJECT_ROOT / "data" / "creditcard.csv"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "models" / "credit_card" / "initial"),
                        help="Where to save trained artifacts. Defaults to models/credit_card/initial/ (gitignored).")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train-windows", type=int, default=0,
                        help="Cap training windows for a fast smoke run (0 = use all).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("LSTM ENCODER-DECODER -- FRAUD BASELINE (step 2a)")
    print("=" * 60)
    print(f"Output directory:   {args.output_dir}")
    print(f"Sequence length:    {args.seq_len} transactions/window")
    print(f"Hidden dim:         {args.hidden_dim}")
    print(f"Max epochs:         {args.epochs}")
    print(f"Device:             {device}")
    print()
    print("This is a fast baseline -- expect a modest PR-AUC. Run")
    print("    python fraud/4_grid_sweep_fraud.py")
    print("next to search for a better configuration and retrain the winner")
    print("into models/credit_card/best/.")
    print("=" * 60)

    # Step 1: Preprocess
    print("\n--- Step 1: Preprocessing transactions ---")
    pp_config = FraudPreprocessorConfig(sequence_length=args.seq_len, stride=args.seq_len)
    window_splits, label_splits, scaler, pp_config = preprocess_pipeline(args.data_path, pp_config)

    train_windows = window_splits["train"]
    if args.max_train_windows and len(train_windows) > args.max_train_windows:
        idx = np.random.choice(len(train_windows), args.max_train_windows, replace=False)
        train_windows = train_windows[idx]
        logger.info(f"Subsampled training windows to {len(train_windows):,} (smoke run)")

    n_features = train_windows.shape[2]
    train_loader = make_loader(train_windows, args.batch_size, shuffle=True)
    val_loader = make_loader(window_splits["val"], args.batch_size, shuffle=False)

    # Step 2: Model (multivariate)
    print("\n--- Step 2: Creating model ---")
    model = create_model(
        input_dim=n_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        sequence_length=args.seq_len,
    )
    model.to(device)
    print(f"Model config: {model.get_config()}")

    # Step 3: Train on legit windows
    print("\n--- Step 3: Training (reconstruction loss, legit windows only) ---")
    train_cfg = TrainingConfig(epochs=args.epochs, learning_rate=args.lr, patience=args.patience)
    model, history = train_model(model, train_loader, val_loader, device, config=train_cfg)

    # Step 4: Fit point-level multivariate scorer on legit training windows
    print("\n--- Step 4: Fitting anomaly scorer (point-level Mahalanobis) ---")
    scorer = AnomalyScorer(ScorerConfig(scoring_mode="point", threshold_method="percentile"))
    scorer.fit(model, train_loader, device)

    # Step 5: Evaluate per-transaction on the test split
    print("\n--- Step 5: Evaluating on test set ---")
    metrics = evaluate(model, scorer, window_splits, label_splits, device, args.batch_size)
    scorer.point_threshold = metrics["threshold"]
    print_metrics(metrics, title="BASELINE TEST RESULTS (per transaction)")

    # Step 6: Save artifacts
    print("\n--- Step 6: Saving artifacts ---")
    save_training_artifacts(
        output_dir=args.output_dir,
        model=model,
        scaler=scaler,
        scorer=scorer,
        history=history,
        preprocess_config=pp_config,
    )
    print(f"\nArtifacts saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
