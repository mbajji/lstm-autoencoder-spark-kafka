"""
Step 2a (CICIDS2017): Train the baseline network-intrusion detector.

Trains an LSTM Encoder-Decoder on benign CICIDS2017 flow windows and saves
artifacts to `models/cicids/initial/`. Mirrors the taxi/fraud baselines: a
fast initial config (hidden_dim=32, seq_len=20, 10 epochs) that learns the
normal-traffic manifold, so attacks (DoS/DDoS/PortScan/...) show up as high
point-level Mahalanobis reconstruction error.

Usage:
    python code/1_train_cicids.py
    python code/1_train_cicids.py --epochs 30 --hidden-dim 64
    python code/1_train_cicids.py --max-rows 300000   # fast smoke run
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_model
from src.scorer import AnomalyScorer, ScorerConfig
from src.preprocess_cicids import CICIDSPreprocessorConfig, preprocess_pipeline
from src.training import TrainingConfig, save_training_artifacts, train_model
from src.fraud_eval import make_loader, evaluate, print_metrics

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train baseline LSTM-AE for CICIDS2017 intrusion detection")
    parser.add_argument("--data-path", type=str, default=str(PROJECT_ROOT / "data" / "cicids2017_cleaned.csv"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "models" / "cicids" / "initial"))
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Cap rows read for a fast smoke run (0 = use all 2.5M).")
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
    print("LSTM ENCODER-DECODER -- CICIDS2017 INTRUSION BASELINE (step 2a)")
    print("=" * 60)
    print(f"Output directory:   {args.output_dir}")
    print(f"Sequence length:    {args.seq_len} flows/window")
    print(f"Hidden dim:         {args.hidden_dim}")
    print(f"Max epochs:         {args.epochs}")
    print(f"Device:             {device}")
    print()
    print("Fast baseline -- run code/4_grid_sweep_cicids.py next to improve it")
    print("and retrain the winner into models/cicids/best/.")
    print("=" * 60)

    print("\n--- Step 1: Preprocessing flows ---")
    pp_config = CICIDSPreprocessorConfig(
        sequence_length=args.seq_len,
        stride=args.seq_len,
        max_rows=args.max_rows or None,
    )
    window_splits, label_splits, scaler, pp_config = preprocess_pipeline(args.data_path, pp_config)

    n_features = window_splits["train"].shape[2]
    train_loader = make_loader(window_splits["train"], args.batch_size, shuffle=True)
    val_loader = make_loader(window_splits["val"], args.batch_size, shuffle=False)

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

    print("\n--- Step 3: Training (reconstruction loss, benign windows only) ---")
    train_cfg = TrainingConfig(epochs=args.epochs, learning_rate=args.lr, patience=args.patience)
    model, history = train_model(model, train_loader, val_loader, device, config=train_cfg)

    print("\n--- Step 4: Fitting anomaly scorer (point-level Mahalanobis) ---")
    scorer = AnomalyScorer(ScorerConfig(scoring_mode="point", threshold_method="percentile"))
    scorer.fit(model, train_loader, device)

    print("\n--- Step 5: Evaluating on test set ---")
    metrics = evaluate(model, scorer, window_splits, label_splits, device, args.batch_size)
    scorer.point_threshold = metrics["threshold"]
    print_metrics(metrics, title="BASELINE TEST RESULTS (per flow)")

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
