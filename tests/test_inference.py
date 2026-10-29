"""
Unit and regression test suite for Financial Sentiment Analysis.

Tests:
1. Text normalization and financial domain preservation in preprocessing.
2. Baseline pipeline serialization and probability calibration.
3. FinBERT inference, GPU/CPU tensor flow, and output schema alignment.
4. Financial probe assertions on canonical bullish, bearish, and disclosure sentences.
5. Unified prediction contract and interpretability feature attribution.
"""

import os
import pytest
import numpy as np

from src.preprocess import clean_text_baseline, clean_text_transformer, expand_contractions
from src.predict import UnifiedPredictor, ID_TO_LABEL


@pytest.fixture(scope="module")
def predictor():
    """Initializes UnifiedPredictor once for the test session."""
    return UnifiedPredictor(config_path="config/config.yaml")


class TestPreprocessing:
    def test_contraction_expansion(self):
        text = "The company didn't meet expectations and won't pay dividends."
        expanded = expand_contractions(text)
        assert "did not" in expanded
        assert "will not" in expanded
        assert "didn't" not in expanded

    def test_baseline_clean_preserves_financial_symbols(self):
        text = "Revenue grew by +14.5% YoY to $350M, beating EBITDA guidance."
        cleaned = clean_text_baseline(text)
        # Should lowercase and preserve %, $, +, -
        assert "%" in cleaned
        assert "$" in cleaned
        assert "14.5" in cleaned
        assert "ebitda" in cleaned
        assert "yoy" in cleaned

    def test_transformer_clean_preserves_casing_and_punctuation(self):
        text = "<b>Alert:</b> Net sales dropped by 12% in Q3; AGM scheduled."
        cleaned = clean_text_transformer(text)
        assert "<b>" not in cleaned
        assert "Alert:" in cleaned
        assert "Q3" in cleaned
        assert "AGM" in cleaned


class TestUnifiedPredictor:
    def test_baseline_predictions_and_probabilities(self, predictor):
        sentences = [
            "Operating profit expanded by 25% year-on-year.",
            "The company announced the date of its annual general meeting.",
        ]
        res = predictor.predict_baseline(sentences)

        assert "labels" in res
        assert "probabilities" in res
        assert len(res["labels"]) == 2
        assert len(res["probabilities"]) == 2

        probs = np.array(res["probabilities"])
        # Ensure probabilities sum to 1.0
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1e-4)

    def test_finbert_predictions_and_probabilities(self, predictor):
        sentences = [
            "Profit before tax surged to 120 million euros.",
            "Net sales declined sharply due to supply chain disruption.",
        ]
        res = predictor.predict_finbert(sentences)

        assert "labels" in res
        assert "probabilities" in res
        assert len(res["labels"]) == 2
        assert len(res["probabilities"]) == 2

        probs = np.array(res["probabilities"])
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1e-4)

    def test_predict_both_schema(self, predictor):
        text = "Operating profit grew to EUR 12.5m from EUR 8.1m."
        records = predictor.predict_both(text)

        assert isinstance(records, list)
        assert len(records) == 1
        record = records[0]

        assert "sentence" in record
        assert "baseline" in record
        assert "finbert" in record
        assert "agreement" in record

        for model_key in ["baseline", "finbert"]:
            assert "label" in record[model_key]
            assert "label_name" in record[model_key]
            assert "confidence" in record[model_key]
            assert "probabilities" in record[model_key]
            assert record[model_key]["label_name"] in ["negative", "neutral", "positive"]

    def test_probe_sentences(self, predictor):
        # 1. Clear Positive
        pos_res = predictor.predict_both("Net sales rose 20% to EUR 210 million, with profit beating expectations.")[0]
        assert pos_res["finbert"]["label_name"] == "positive"
        assert pos_res["baseline"]["label_name"] == "positive"

        # 2. Clear Neutral disclosure
        neu_res = predictor.predict_both("The board of directors convened on Monday at company headquarters.")[0]
        assert neu_res["finbert"]["label_name"] == "neutral"
        assert neu_res["baseline"]["label_name"] == "neutral"

        # 3. Contextual Negative distress
        neg_res = predictor.predict_both("Operating profit fell by 60% due to severe cost escalation and weak demand.")[0]
        assert neg_res["finbert"]["label_name"] == "negative"

    def test_baseline_feature_contributions(self, predictor):
        text = "Net profit rose substantially to record levels."
        contribs = predictor.get_baseline_feature_contributions(text)

        assert "active_tokens" in contribs
        assert "class_contributions" in contribs
        assert "positive" in contribs["class_contributions"]
        # Words like 'rose' or 'profit' should be active
        active = " ".join(contribs["active_tokens"])
        assert "profit" in active or "rose" in active

