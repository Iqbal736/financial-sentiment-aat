"""
Preprocessing module for financial sentiment classification.

Financial text differs significantly from standard sentiment datasets:
- Numbers, percentages, and currencies carry polarity (e.g., "grew 14%", "fell 8%").
- Financial acronyms (YoY, QoQ, EBITDA, EPS, bps, AGM) carry domain meaning.
- Contractions (e.g., didn't, haven't) must be expanded to preserve negative polarity.
- TF-IDF requires lowercasing and selective symbol retention, while FinBERT
  benefits from cased natural punctuation and raw token flows.
"""

import html
import re
from typing import Union
import pandas as pd

# Comprehensive dictionary for contraction expansion
CONTRACTIONS_DICT = {
    r"\bcan't\b": "cannot",
    r"\bcant\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bwont\b": "will not",
    r"\bdidn't\b": "did not",
    r"\bdidnt\b": "did not",
    r"\bdoesn't\b": "does not",
    r"\bdoesnt\b": "does not",
    r"\bdon't\b": "do not",
    r"\bdont\b": "do not",
    r"\bisn't\b": "is not",
    r"\bisnt\b": "is not",
    r"\baren't\b": "are not",
    r"\barent\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bwasnt\b": "was not",
    r"\bweren't\b": "were not",
    r"\bwerent\b": "were not",
    r"\bhasn't\b": "has not",
    r"\bhasnt\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhavent\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bhadnt\b": "had not",
    r"\bwouldn't\b": "would not",
    r"\bwouldnt\b": "would not",
    r"\bshouldn't\b": "should not",
    r"\bshouldnt\b": "should not",
    r"\bcouldn't\b": "could not",
    r"\bcouldnt\b": "could not",
    r"\bit's\b": "it is",
    r"\bthere's\b": "there is",
    r"\bthat's\b": "that is",
    r"\bwhat's\b": "what is",
    r"\bwho's\b": "who is",
    r"\bi'm\b": "i am",
    r"\byou're\b": "you are",
    r"\bwe're\b": "we are",
    r"\bthey're\b": "they are",
    r"\bi've\b": "i have",
    r"\byou've\b": "you have",
    r"\bwe've\b": "we have",
    r"\bthey've\b": "they have",
    r"\bi'll\b": "i will",
    r"\byou'll\b": "you will",
    r"\bhe'll\b": "he will",
    r"\bshe'll\b": "she will",
    r"\bwe'll\b": "we will",
    r"\bthey'll\b": "they will",
}

# Compile patterns for fast processing
CONTRACTIONS_COMPILED = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in CONTRACTIONS_DICT.items()
]

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")

# In TF-IDF, preserve alphanumeric characters, currencies, and percentage symbols:
# $ € £ ¥ ₹ % + - . /
TFIDF_TOKEN_FILTER_PATTERN = re.compile(r"[^\w\s\$\€\£\¥\₹\%\+\-\.\/]")


def expand_contractions(text: str) -> str:
    """Expand conversational contractions to ensure negation words ('not') are preserved."""
    for pattern, replacement in CONTRACTIONS_COMPILED:
        text = pattern.sub(replacement, text)
    return text


def strip_html_tags(text: str) -> str:
    """Remove HTML formatting and unescape entities."""
    text = html.unescape(text)
    return HTML_TAG_PATTERN.sub(" ", text)


def clean_text_baseline(text: str) -> str:
    """
    Preprocessing pipeline for TF-IDF + Logistic Regression:
    1. Expand contractions (e.g. didn't -> did not).
    2. Strip rogue HTML.
    3. Lowercase text.
    4. Retain alphanumeric words, numbers, and critical financial symbols ($, %, etc.).
    5. Collapse repeated whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = strip_html_tags(text)
    text = expand_contractions(text)
    text = text.lower()
    text = TFIDF_TOKEN_FILTER_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def clean_text_transformer(text: str) -> str:
    """
    Minimal preprocessing for FinBERT / Transformer:
    1. Strip rogue HTML and unescape HTML characters.
    2. Normalize non-breaking and strange unicode whitespace.
    3. Retain natural casing, punctuation, numbers, and symbols since BERT's WordPiece
       tokenizer utilizes casing and punctuation context.
    """
    if not isinstance(text, str):
        return ""
    text = strip_html_tags(text)
    # Replace non-breaking spaces and zero-width spaces
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def preprocess_series(series: pd.Series, model_type: str = "baseline") -> pd.Series:
    """Apply cleaning over a Pandas Series based on the model type."""
    if model_type.lower() in ("finbert", "transformer", "bert"):
        return series.astype(str).apply(clean_text_transformer)
    return series.astype(str).apply(clean_text_baseline)

