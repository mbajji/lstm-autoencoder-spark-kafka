"""
Step 2b / 2d (CICIDS2017): Evaluate a trained intrusion detector.

Reads a saved artifact directory (default `models/cicids/initial/`, or pass
`--model-dir models/cicids/best`), re-runs preprocessing with the SAME split
config that produced it, reprints the per-flow metrics, and writes diagnostic
plots to `evaluation/`:

    - cicids_score_distribution.png : benign vs attack point-score histograms
    - cicids_pr_curve.png           : precision-recall curve (PR-AUC headline)

Usage:
    python cicids/2_evaluate_cicids.py
    python cicids/2_evaluate_cicids.py --model-dir models/cicids/best
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import EncDecAD
from src.scorer import AnomalyScorer
from src.preprocess_cicids import CICIDSPreprocessorConfig, preprocess_pipeline
from src.fraud_eval import evaluate, print_metrics

logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device) -> EncDecAD:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = EncDecAD(config=checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    return model


def plot_score_distribution(scores, labels, threshold, out_path):
    benign = scores[labels == 0]
    attack = scores[labels == 1]
    plt.figure(figsize=(10, 5))
    lo = max(min(scores.min(), benign.min() if len(benign) else 1.0), 1e-3)
    bins = np.logspace(np.log10(lo), np.log10(scores.max() + 1), 80)
    plt.hist(benign, bins=bins, alpha=0.6, label=f"benign (n={len(benign):,})", color="#4c72b0", density=True)
    plt.hist(attack, bins=bins, alpha=0.7, label=f"attack (n={len(attack):,})", color="#c44e52", density=True)
    plt.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.1f}")
    plt.xscale("log")
    plt.xlabel("Point-level Mahalanobis score")
    plt.ylabel("density")
    plt.title("Anomaly score distribution: benign vs attack (CICIDS2017)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    logger.info(f"Wrote {out_path}")


def plot_pr_curve(scores, labels, pr_auc, out_path):
    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="#c44e52", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    baseline = labels.mean()
    plt.axhline(baseline, color="gray", linestyle="--", label=f"random ({baseline:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve (per flow, CICIDS2017)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    logger.info(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CICIDS2017 LSTM-AE model")
    parser.add_argument("--data-path", type=str, default=str(PROJECT_ROOT / "data" / "cicids2017_cleaned.csv"))
    parser.add_argument("--model-dir", type=str, default=str(PROJECT_ROOT / "models" / "cicids" / "initial"))
    parser.add_argument("--eval-dir", type=str, default=str(PROJECT_ROOT / "evaluation"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    model_dir = Path(args.model_dir)
    if not (model_dir / "lstm_model.pt").exists():
        raise FileNotFoundError(
            f"No model at {model_dir}. Run cicids/1_train_cicids.py "
            f"(or 4_grid_sweep_cicids.py) first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print(f"EVALUATING CICIDS2017 MODEL: {model_dir}")
    print("=" * 60)

    with open(model_dir / "preprocessor_config.pkl", "rb") as f:
        pp_config: CICIDSPreprocessorConfig = pickle.load(f)

    window_splits, label_splits, _, _ = preprocess_pipeline(args.data_path, pp_config)

    model = load_model(str(model_dir / "lstm_model.pt"), device)
    scorer = AnomalyScorer.load(str(model_dir / "scorer.pkl"))

    metrics = evaluate(model, scorer, window_splits, label_splits, device, args.batch_size)
    print_metrics(metrics, title=f"TEST RESULTS -- {model_dir.name} (per flow)")

    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    plot_score_distribution(
        metrics["test_scores"], metrics["test_labels"], metrics["threshold"],
        eval_dir / "cicids_score_distribution.png",
    )
    plot_pr_curve(
        metrics["test_scores"], metrics["test_labels"], metrics["pr_auc"],
        eval_dir / "cicids_pr_curve.png",
    )

    print(f"\nPlots written to: {eval_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
