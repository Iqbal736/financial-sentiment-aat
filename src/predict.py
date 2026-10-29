"""
Unified Inference Module for Financial Sentiment Analysis.

Provides UnifiedPredictor to query both:
1. Classical Baseline: TF-IDF + Multinomial Logistic Regression
2. Transformer: ProsusAI/finbert (Domain-adapted BERT)

Standardized Project Labels:
  0 -> Negative
  1 -> Neutral
  2 -> Positive
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import yaml

from src.preprocess import clean_text_baseline, clean_text_transformer

ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}


class UnifiedPredictor:
    """
    Unified predictor supporting both TF-IDF Logistic Regression and FinBERT.
    Handles device placement, tokenization, inference latency measurement,
    and output schema normalization.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        device: Optional[str] = None,
        load_finbert: bool = True,
        load_baseline: bool = True,
    ):
        self.config = self._load_config(config_path)

        # Device determination
        configured_device = self.config.get("finbert", {}).get("device", "auto")
        if device is not None:
            self.device_str = device
        elif configured_device == "auto":
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = configured_device
        self.device = torch.device(self.device_str)

        self.baseline_pipeline = None
        self.finbert_model = None
        self.finbert_tokenizer = None

        if load_baseline:
            self._load_baseline_pipeline()

        if load_finbert:
            self._load_finbert_model()

    def _load_config(self, config_path: str) -> dict:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def _load_baseline_pipeline(self) -> None:
        model_path = self.config.get(
            "baseline", {}
        ).get("model_path", "models/tfidf_logistic.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Baseline model not found at {model_path}. Run train_baseline.py first."
            )
        self.baseline_pipeline = joblib.load(model_path)

    def _load_finbert_model(self) -> None:
        model_name = self.config.get("finbert", {}).get("model_name", "ProsusAI/finbert")
        self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.finbert_model.to(self.device)
        self.finbert_model.eval()

        # FinBERT raw mapping is: 0 -> positive, 1 -> negative, 2 -> neutral
        # Our standard project mapping is: 0 -> negative, 1 -> neutral, 2 -> positive
        # Construct index permutation vector: [raw_idx_for_neg, raw_idx_for_neu, raw_idx_for_pos]
        raw_id2label = {int(k): v.lower() for k, v in self.finbert_model.config.id2label.items()}
        raw_label2id = {v: k for k, v in raw_id2label.items()}
        self.finbert_reorder_indices = [
            raw_label2id["negative"],  # Project index 0
            raw_label2id["neutral"],   # Project index 1
            raw_label2id["positive"],  # Project index 2
        ]

    def _format_input(self, texts: Union[str, List[str]]) -> List[str]:
        if isinstance(texts, str):
            return [texts]
        return list(texts)

    def predict_baseline(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Runs TF-IDF + Logistic Regression inference.
        Returns:
            dict containing labels, label_names, probabilities, latency_ms.
        """
        if self.baseline_pipeline is None:
            self._load_baseline_pipeline()

        text_list = self._format_input(texts)
        start_time = time.perf_counter()

        # Pipeline applies clean_text_baseline inside TfidfVectorizer
        probs = self.baseline_pipeline.predict_proba(text_list)
        preds = np.argmax(probs, axis=1)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        label_names = [ID_TO_LABEL[int(p)] for p in preds]

        return {
            "model": "TF-IDF + Logistic Regression",
            "labels": preds.tolist(),
            "label_names": label_names,
            "probabilities": probs.tolist(),  # Shape: (N, 3) [Negative, Neutral, Positive]
            "latency_ms_total": round(elapsed_ms, 3),
            "latency_ms_per_sample": round(elapsed_ms / max(1, len(text_list)), 3),
        }

    def predict_finbert(
        self, texts: Union[str, List[str]], max_length: int = 128, batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Runs FinBERT inference with automatic batching and GPU acceleration.
        Returns:
            dict containing labels, label_names, probabilities, latency_ms.
        """
        if self.finbert_model is None or self.finbert_tokenizer is None:
            self._load_finbert_model()

        text_list = self._format_input(texts)
        cleaned_texts = [clean_text_transformer(t) for t in text_list]

        all_probs = []
        start_time = time.perf_counter()

        with torch.no_grad():
            for i in range(0, len(cleaned_texts), batch_size):
                batch = cleaned_texts[i : i + batch_size]
                encoded = self.finbert_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.finbert_model(**encoded)
                raw_probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

                # Reorder columns to align with project standard: [Negative, Neutral, Positive]
                standardized_probs = raw_probs[:, self.finbert_reorder_indices]
                all_probs.append(standardized_probs)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        final_probs = np.vstack(all_probs)
        preds = np.argmax(final_probs, axis=1)
        label_names = [ID_TO_LABEL[int(p)] for p in preds]

        return {
            "model": "ProsusAI/finbert",
            "labels": preds.tolist(),
            "label_names": label_names,
            "probabilities": final_probs.tolist(),
            "latency_ms_total": round(elapsed_ms, 3),
            "latency_ms_per_sample": round(elapsed_ms / max(1, len(text_list)), 3),
            "device": str(self.device),
        }

    def predict_both(
        self, texts: Union[str, List[str]], max_length: int = 128
    ) -> List[Dict[str, Any]]:
        """
        Runs both models on the input text(s) and returns unified comparison records.
        """
        text_list = self._format_input(texts)
        baseline_res = self.predict_baseline(text_list)
        finbert_res = self.predict_finbert(text_list, max_length=max_length)

        results = []
        for i, text in enumerate(text_list):
            b_label = baseline_res["labels"][i]
            f_label = finbert_res["labels"][i]
            b_probs = baseline_res["probabilities"][i]
            f_probs = finbert_res["probabilities"][i]

            results.append({
                "sentence": text,
                "baseline": {
                    "label": b_label,
                    "label_name": ID_TO_LABEL[b_label],
                    "confidence": round(float(b_probs[b_label]), 4),
                    "probabilities": {
                        "negative": round(float(b_probs[0]), 4),
                        "neutral": round(float(b_probs[1]), 4),
                        "positive": round(float(b_probs[2]), 4),
                    },
                },
                "finbert": {
                    "label": f_label,
                    "label_name": ID_TO_LABEL[f_label],
                    "confidence": round(float(f_probs[f_label]), 4),
                    "probabilities": {
                        "negative": round(float(f_probs[0]), 4),
                        "neutral": round(float(f_probs[1]), 4),
                        "positive": round(float(f_probs[2]), 4),
                    },
                },
                "agreement": bool(b_label == f_label),
            })
        return results

    def get_baseline_feature_contributions(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Extracts TF-IDF features and logistic regression log-odds weights
        for the given text to explain why the baseline predicted its label.
        """
        if self.baseline_pipeline is None:
            self._load_baseline_pipeline()

        vectorizer = self.baseline_pipeline.named_steps["tfidf"]
        classifier = self.baseline_pipeline.named_steps["clf"]

        cleaned = clean_text_baseline(text)
        vec = vectorizer.transform([cleaned])
        feature_names = np.array(vectorizer.get_feature_names_out())

        nonzero_indices = vec.indices
        if len(nonzero_indices) == 0:
            return {"active_features": [], "top_positive": [], "top_negative": []}

        tfidf_values = vec.data
        words = feature_names[nonzero_indices]

        # Classifier coef shape: (3, n_features)
        # Class 0: Negative, Class 1: Neutral, Class 2: Positive
        contributions = {}
        for class_idx, class_name in ID_TO_LABEL.items():
            coefs = classifier.coef_[class_idx, nonzero_indices]
            scores = tfidf_values * coefs
            ranked_idx = np.argsort(scores)[::-1]
            contributions[class_name] = [
                {"token": words[j], "score": round(float(scores[j]), 4)}
                for j in ranked_idx[:top_k]
                if scores[j] > 0
            ]

        return {
            "cleaned_text": cleaned,
            "active_tokens": words.tolist(),
            "class_contributions": contributions,
        }

