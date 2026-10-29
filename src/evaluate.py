"""
Comprehensive Evaluation Module.

Computes:
1. Classification metrics: Accuracy, Macro/Weighted/Per-Class F1, Precision, Recall.
2. Computational footprint: Parameter counts, disk footprint, inference latency & throughput.
3. McNemar's Statistical Significance Test on paired test set predictions.
4. Export of results/metrics.json, results/confusion_matrices.png, and results/error_analysis.json.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import yaml

from src.predict import ID_TO_LABEL, UnifiedPredictor


def load_config(config_path: str = "config/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def compute_mcnemar_test(
    y_true: np.ndarray, y_pred_baseline: np.ndarray, y_pred_finbert: np.ndarray
) -> Dict[str, Any]:
    """
    Computes McNemar's Test for paired nominal data comparing two classifiers.
    Contingency Table:
        n00: Both correct
        n01: Baseline correct, FinBERT incorrect
        n10: Baseline incorrect, FinBERT correct
        n11: Both incorrect
    Statistic: chi2 = (|n01 - n10| - 1)^2 / (n01 + n10) with Edwards' continuity correction.
    """
    b_correct = (y_pred_baseline == y_true)
    f_correct = (y_pred_finbert == y_true)

    n00 = int(np.sum(b_correct & f_correct))
    n01 = int(np.sum(b_correct & ~f_correct))
    n10 = int(np.sum(~b_correct & f_correct))
    n11 = int(np.sum(~b_correct & ~f_correct))

    discordant = n01 + n10
    if discordant == 0:
        chi2_stat = 0.0
        p_val = 1.0
    else:
        # Edwards' continuity correction
        chi2_stat = float(((abs(n01 - n10) - 1.0) ** 2) / discordant)
        p_val = float(1.0 - stats.chi2.cdf(chi2_stat, df=1))

    odds_ratio = float(n10 / n01) if n01 > 0 else float("inf")

    return {
        "contingency_table": {
            "both_correct (n00)": n00,
            "baseline_correct_finbert_incorrect (n01)": n01,
            "baseline_incorrect_finbert_correct (n10)": n10,
            "both_incorrect (n11)": n11,
        },
        "mcnemar_chi2": round(chi2_stat, 4),
        "p_value": round(p_val, 6),
        "odds_ratio": round(odds_ratio, 4) if odds_ratio != float("inf") else "inf",
        "statistically_significant (alpha=0.05)": bool(p_val < 0.05),
        "interpretation": (
            "The performance difference between FinBERT and TF-IDF Baseline is statistically "
            f"significant (p={p_val:.5f} < 0.05)."
            if p_val < 0.05
            else "The performance difference is not statistically significant at alpha=0.05."
        ),
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes overall and per-class precision, recall, and f1 scores."""
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    weighted_p = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_r = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class = {}
    for idx, name in ID_TO_LABEL.items():
        mask_true = (y_true == idx)
        mask_pred = (y_pred == idx)
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        support = int(np.sum(mask_true))

        per_class[name] = {
            "class_id": idx,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_avg": {
            "precision": round(float(macro_p), 4),
            "recall": round(float(macro_r), 4),
            "f1": round(float(macro_f1), 4),
        },
        "weighted_avg": {
            "precision": round(float(weighted_p), 4),
            "recall": round(float(weighted_r), 4),
            "f1": round(float(weighted_f1), 4),
        },
        "per_class": per_class,
    }


def plot_and_save_confusion_matrices(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_finbert: np.ndarray,
    output_path: str = "results/confusion_matrices.png",
) -> None:
    """Generates dual normalized confusion matrix heatmaps."""
    class_names = ["Negative", "Neutral", "Positive"]
    cm_base = confusion_matrix(y_true, y_pred_baseline, labels=[0, 1, 2])
    cm_fin = confusion_matrix(y_true, y_pred_finbert, labels=[0, 1, 2])

    cm_base_norm = cm_base.astype("float") / cm_base.sum(axis=1)[:, np.newaxis]
    cm_fin_norm = cm_fin.astype("float") / cm_fin.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    plt.subplots_adjust(wspace=0.3)

    # Format annotations with count + percentage
    def get_annot(cm, cm_norm):
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
        return annot

    annot_base = get_annot(cm_base, cm_base_norm)
    annot_fin = get_annot(cm_fin, cm_fin_norm)

    sns.heatmap(
        cm_base_norm,
        annot=annot_base,
        fmt="",
        cmap="Blues",
        ax=axes[0],
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title(
        "TF-IDF + Logistic Regression Baseline\n(Confusion Matrix)", fontsize=13, pad=12, fontweight="bold"
    )
    axes[0].set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("True Label", fontsize=11, fontweight="bold")

    sns.heatmap(
        cm_fin_norm,
        annot=annot_fin,
        fmt="",
        cmap="Greens",
        ax=axes[1],
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title(
        "ProsusAI/finbert (Domain Transformer)\n(Confusion Matrix)", fontsize=13, pad=12, fontweight="bold"
    )
    axes[1].set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("True Label", fontsize=11, fontweight="bold")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrices plot saved to: {output_path}")


def analyze_errors_and_disagreements(
    df: pd.DataFrame,
    y_pred_baseline: np.ndarray,
    y_probs_baseline: np.ndarray,
    y_pred_finbert: np.ndarray,
    y_probs_finbert: np.ndarray,
    output_path: str = "results/error_analysis.json",
) -> List[Dict[str, Any]]:
    """Analyzes disagreement cases between models and baseline errors."""
    y_true = df["label"].values
    sentences = df["sentence"].values

    records = []
    disagreements = []

    for i in range(len(df)):
        true_lbl = int(y_true[i])
        b_lbl = int(y_pred_baseline[i])
        f_lbl = int(y_pred_finbert[i])

        rec = {
            "index": i,
            "sentence": sentences[i],
            "true_label": true_lbl,
            "true_label_name": ID_TO_LABEL[true_lbl],
            "baseline": {
                "pred_label": b_lbl,
                "pred_label_name": ID_TO_LABEL[b_lbl],
                "confidence": round(float(y_probs_baseline[i][b_lbl]), 4),
                "is_correct": bool(b_lbl == true_lbl),
                "probabilities": {
                    "negative": round(float(y_probs_baseline[i][0]), 4),
                    "neutral": round(float(y_probs_baseline[i][1]), 4),
                    "positive": round(float(y_probs_baseline[i][2]), 4),
                },
            },
            "finbert": {
                "pred_label": f_lbl,
                "pred_label_name": ID_TO_LABEL[f_lbl],
                "confidence": round(float(y_probs_finbert[i][f_lbl]), 4),
                "is_correct": bool(f_lbl == true_lbl),
                "probabilities": {
                    "negative": round(float(y_probs_finbert[i][0]), 4),
                    "neutral": round(float(y_probs_finbert[i][1]), 4),
                    "positive": round(float(y_probs_finbert[i][2]), 4),
                },
            },
            "models_agree": bool(b_lbl == f_lbl),
        }

        # Classify the type of error / divergence
        if b_lbl != f_lbl:
            if f_lbl == true_lbl and b_lbl != true_lbl:
                category = "FinBERT Correct, Baseline Failed (Contextual Financial Nuance)"
            elif b_lbl == true_lbl and f_lbl != true_lbl:
                category = "Baseline Correct, FinBERT Failed"
            else:
                category = "Both Failed, Different Errors"
            rec["disagreement_category"] = category
            disagreements.append(rec)
        else:
            rec["disagreement_category"] = "Consensus"

        records.append(rec)

    output_data = {
        "summary": {
            "total_test_samples": len(df),
            "agreements_count": len(df) - len(disagreements),
            "disagreements_count": len(disagreements),
            "agreement_rate": round((len(df) - len(disagreements)) / len(df), 4),
            "finbert_only_correct": sum(
                1 for d in disagreements if d["finbert"]["is_correct"] and not d["baseline"]["is_correct"]
            ),
            "baseline_only_correct": sum(
                1 for d in disagreements if d["baseline"]["is_correct"] and not d["finbert"]["is_correct"]
            ),
            "both_incorrect_count": sum(
                1 for d in disagreements if not d["baseline"]["is_correct"] and not d["finbert"]["is_correct"]
            ),
        },
        "disagreements": disagreements,
        "all_samples": records,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Error analysis saved to: {output_path}")

    return disagreements


def run_evaluation(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Runs end-to-end evaluation and exports metrics, plots, and error reports."""
    cfg = load_config(config_path)
    test_path = cfg.get("data", {}).get("test_path", "data/test.csv")
    metrics_path = cfg.get("evaluation", {}).get("metrics_path", "results/metrics.json")
    cm_path = cfg.get("evaluation", {}).get("confusion_matrices_path", "results/confusion_matrices.png")
    err_path = cfg.get("evaluation", {}).get("error_analysis_path", "results/error_analysis.json")

    print(f"Loading held-out test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    y_true = test_df["label"].values
    sentences = test_df["sentence"].tolist()
    print(f"Loaded {len(test_df)} test sentences.")

    print("\nInitializing UnifiedPredictor...")
    predictor = UnifiedPredictor(config_path=config_path)

    print("\n[1/3] Running TF-IDF + Logistic Regression inference...")
    t0 = time.perf_counter()
    b_res = predictor.predict_baseline(sentences)
    b_time = time.perf_counter() - t0
    y_pred_b = np.array(b_res["labels"])
    y_probs_b = np.array(b_res["probabilities"])

    print(f"[2/3] Running FinBERT inference on device: {predictor.device}...")
    t0 = time.perf_counter()
    f_res = predictor.predict_finbert(sentences, batch_size=32)
    f_time = time.perf_counter() - t0
    y_pred_f = np.array(f_res["labels"])
    y_probs_f = np.array(f_res["probabilities"])

    print("\nComputing classification metrics...")
    b_metrics = compute_classification_metrics(y_true, y_pred_b)
    f_metrics = compute_classification_metrics(y_true, y_pred_f)

    # Compute parameter footprint and model sizes
    baseline_model_path = cfg.get("baseline", {}).get("model_path", "models/tfidf_logistic.joblib")
    b_file_size_mb = os.path.getsize(baseline_model_path) / (1024 * 1024) if os.path.exists(baseline_model_path) else 0.0
    b_vocab_size = len(predictor.baseline_pipeline.named_steps["tfidf"].vocabulary_)
    b_param_count = predictor.baseline_pipeline.named_steps["clf"].coef_.size + predictor.baseline_pipeline.named_steps["clf"].intercept_.size

    f_total_params = sum(p.numel() for p in predictor.finbert_model.parameters())
    f_trainable_params = sum(p.numel() for p in predictor.finbert_model.parameters() if p.requires_grad)

    mcnemar_results = compute_mcnemar_test(y_true, y_pred_b, y_pred_f)

    metrics_payload = {
        "dataset": {
            "test_samples": len(test_df),
            "class_distribution": {
                name: int(np.sum(y_true == idx)) for idx, name in ID_TO_LABEL.items()
            },
        },
        "baseline_model": {
            "name": "TF-IDF (1-2 ngrams, sublinear) + Logistic Regression",
            "metrics": b_metrics,
            "footprint": {
                "vocabulary_features": b_vocab_size,
                "parameter_count": b_param_count,
                "disk_size_mb": round(b_file_size_mb, 2),
            },
            "latency": {
                "total_time_seconds": round(b_time, 4),
                "ms_per_sample": round((b_time / len(test_df)) * 1000.0, 3),
                "throughput_samples_per_sec": round(len(test_df) / b_time, 2),
            },
        },
        "finbert_model": {
            "name": "ProsusAI/finbert (Domain-adapted BERT)",
            "device": str(predictor.device),
            "metrics": f_metrics,
            "footprint": {
                "total_parameters": f_total_params,
                "trainable_parameters": f_trainable_params,
                "parameter_count_millions": round(f_total_params / 1e6, 2),
                "estimated_weights_disk_mb": 438.0,
            },
            "latency": {
                "total_time_seconds": round(f_time, 4),
                "ms_per_sample": round((f_time / len(test_df)) * 1000.0, 3),
                "throughput_samples_per_sec": round(len(test_df) / f_time, 2),
            },
        },
        "mcnemar_significance_test": mcnemar_results,
    }

    # Save metrics.json
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # Plot Confusion Matrices
    print("\n[3/3] Generating and saving confusion matrix heatmaps...")
    plot_and_save_confusion_matrices(y_true, y_pred_b, y_pred_f, output_path=cm_path)

    # Perform error and disagreement analysis
    analyze_errors_and_disagreements(
        test_df, y_pred_b, y_probs_b, y_pred_f, y_probs_f, output_path=err_path
    )

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Metric                 Baseline (TF-IDF+LR)     FinBERT (Transformer)")
    print(f"Accuracy               {b_metrics['accuracy']:.4f}                   {f_metrics['accuracy']:.4f}")
    print(f"Macro F1               {b_metrics['macro_avg']['f1']:.4f}                   {f_metrics['macro_avg']['f1']:.4f}")
    print(f"Weighted F1            {b_metrics['weighted_avg']['f1']:.4f}                   {f_metrics['weighted_avg']['f1']:.4f}")
    print(f"Latency (ms/sample)    {metrics_payload['baseline_model']['latency']['ms_per_sample']:.2f} ms                  {metrics_payload['finbert_model']['latency']['ms_per_sample']:.2f} ms")
    print(f"Throughput (samples/s) {metrics_payload['baseline_model']['latency']['throughput_samples_per_sec']:.1f}                   {metrics_payload['finbert_model']['latency']['throughput_samples_per_sec']:.1f}")
    print(f"McNemar p-value        {mcnemar_results['p_value']:.6f} (Significant: {mcnemar_results['statistically_significant (alpha=0.05)']})")
    print("====================================================\n")

    return metrics_payload


if __name__ == "__main__":
    run_evaluation()

