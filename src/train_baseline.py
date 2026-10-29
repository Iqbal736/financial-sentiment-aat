"""
Baseline training module for financial sentiment analysis.
Tunes regularisation strength C over a validation grid, evaluates Macro-F1,
retrains on Train + Validation, and exports the serialized pipeline.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
import yaml

from src.preprocess import clean_text_baseline


def load_config(config_path: str = "config/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def train_baseline(config_path: str = "config/config.yaml") -> Tuple[Pipeline, float, dict]:
    """
    Trains the classical baseline:
    - Preprocessing with domain-specific contraction expansion and symbol retention.
    - TF-IDF with sublinear scaling and (1, 2) n-grams.
    - Multinomial Logistic Regression tuned over C parameter search grid.
    - Exports serialized pipeline to models/tfidf_logistic.joblib.
    """
    cfg = load_config(config_path)

    train_path = cfg.get("data", {}).get("train_path", "data/train.csv")
    val_path = cfg.get("data", {}).get("val_path", "data/val.csv")
    model_save_path = Path(cfg.get("baseline", {}).get("model_path", "models/tfidf_logistic.joblib"))
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    c_grid = cfg.get("baseline", {}).get("c_search_grid", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    ngram_range = tuple(cfg.get("baseline", {}).get("tfidf", {}).get("ngram_range", [1, 2]))
    max_features = cfg.get("baseline", {}).get("tfidf", {}).get("max_features", 10000)
    sublinear_tf = cfg.get("baseline", {}).get("tfidf", {}).get("sublinear_tf", True)
    max_iter = cfg.get("baseline", {}).get("logistic_regression", {}).get("max_iter", 1000)
    seed = cfg.get("project", {}).get("seed", 42)

    print(f"Loading training data from {train_path} and validation data from {val_path}...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    best_c = 1.0
    best_f1 = -1.0
    tuning_results: Dict[float, float] = {}

    print(f"\nTuning regularisation parameter C over grid: {c_grid}")
    for c_val in c_grid:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                preprocessor=clean_text_baseline,
                ngram_range=ngram_range,
                max_features=max_features,
                sublinear_tf=sublinear_tf,
                token_pattern=r"(?u)\b\w+\b|[\$\%\€\£\¥\₹]",
            )),
            ("clf", LogisticRegression(
                C=c_val,
                max_iter=max_iter,
                random_state=seed,
                solver="lbfgs",
            )),
        ])

        pipe.fit(train_df["sentence"], train_df["label"])
        val_preds = pipe.predict(val_df["sentence"])
        macro_f1 = f1_score(val_df["label"], val_preds, average="macro")
        tuning_results[c_val] = float(macro_f1)
        print(f"  C={c_val:5.2f} -> Validation Macro-F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_c = c_val

    print(f"\n[Optimal Selection] Best C={best_c} with Validation Macro-F1={best_f1:.4f}")

    # Retrain on combined Train + Validation
    full_train = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Retraining final baseline pipeline on {len(full_train)} combined samples...")

    final_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text_baseline,
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            token_pattern=r"(?u)\b\w+\b|[\$\%\€\£\¥\₹]",
        )),
        ("clf", LogisticRegression(
            C=best_c,
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
        )),
    ])

    final_pipe.fit(full_train["sentence"], full_train["label"])

    joblib.dump(final_pipe, str(model_save_path))
    print(f"Serialized baseline pipeline exported to: {model_save_path}")

    # Compute quick fit metrics
    train_preds = final_pipe.predict(full_train["sentence"])
    report = classification_report(
        full_train["label"],
        train_preds,
        target_names=["Negative (0)", "Neutral (1)", "Positive (2)"],
        output_dict=True,
    )
    print(f"Combined Train Accuracy: {report['accuracy']:.4f}")

    return final_pipe, best_c, tuning_results


if __name__ == "__main__":
    train_baseline()

