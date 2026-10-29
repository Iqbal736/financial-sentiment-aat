# Sentiment Classification of Financial News and Earnings Disclosures

> **Comparative Evaluation:** Classical ML Baseline (TF-IDF + Logistic Regression) vs. Domain-Specific Transformer (`ProsusAI/finbert`).

---

## 1. Project Overview & Scope
* **Official Title:** Sentiment Classification of Financial News and Earnings Disclosures using TF-IDF + Logistic Regression and FinBERT
* **NLP Task:** 3-Class Supervised Text Classification:
  * `0`: **Negative** (Bearish disclosures, warnings, impairments, losses)
  * `1`: **Neutral** (Regulatory announcements, AGM schedules, operational statements)
  * `2`: **Positive** (Revenue growth, dividend declarations, EBITDA expansion)
* **Dataset:** Malo et al. (2014) Financial PhraseBank (`sentences_75agree` subset, 3,453 sentences; Stratified 70% Train, 15% Val, 15% Test).

### The Core Research Question
> *Why do general bag-of-words models fail on financial syntax (where words like "liability", "depreciation", or "margin compression" are contextual rather than emotional), and how much measurable accuracy does contextual domain pre-training add relative to inference latency and parameter footprint?*

---

## 2. Directory Structure

```text
financial-sentiment-aat/
├── config/
│   └── config.yaml             # Seeds, filepaths, C search grid, max_length=128
├── data/
│   ├── raw_phrasebank.csv      # Full downloaded Financial PhraseBank dataset
│   ├── train.csv               # 70% Stratified training set (2,417 sentences)
│   ├── val.csv                 # 15% Stratified validation set (518 sentences)
│   └── test.csv                # 15% Stratified held-out test set (518 sentences)
├── models/
│   └── tfidf_logistic.joblib   # Serialized baseline pipeline
├── results/
│   ├── metrics.json            # Accuracy, F1, Precision, Recall, Latency, McNemar test
│   ├── confusion_matrices.png  # High-res dual confusion matrix heatmaps
│   └── error_analysis.json     # Per-sample predictions, disagreements & error taxonomy
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Loads Financial PhraseBank from Hugging Face
│   ├── preprocess.py           # Text normalisation & cleaning functions
│   ├── train_baseline.py       # GridSearch for C, TF-IDF fit, joblib export
│   ├── evaluate.py             # Computes metrics, latency, and McNemar test
│   └── predict.py              # Unified class returning predictions from both models
├── app.py                      # Multi-page interactive Streamlit application
├── requirements.txt
└── tests/
    └── test_inference.py       # Pytest assertions and financial probe tests
```

---

## 3. Mathematical Foundations

### A. TF-IDF Representation (Sublinear Scaling)
$$\text{TF}(t, d) = 1 + \log(\text{count}(t, d)) \quad \text{if } \text{count}(t, d) > 0, \quad \text{else } 0$$
$$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$
$$w(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$
Each document vector is $L_2$-normalized: $v_{\text{norm}} = \frac{v}{\Vert{}v\Vert{}_2}$.

### B. Multinomial Logistic Regression ($K=3$)
$$P(y = k \mid x) = \frac{\exp(w_k^T x + b_k)}{\sum_{j=1}^K \exp(w_j^T x + b_j)}$$
Minimizing regularized cross-entropy loss with tuned parameter $C$:
$$\mathcal{L}(W) = \frac{1}{2}\Vert{}W\Vert{}_F^2 + C \sum_{i=1}^M \sum_{k=1}^K -y_{ik} \log(P(y_i = k \mid x_i))$$

### C. Scaled Dot-Product Attention in FinBERT
Given Query ($Q$), Key ($K$), and Value ($V$) projections with head dimension $d_k = 64$:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_{12})W^O$$
Final prediction from the pooled $[CLS]$ token hidden state $h_{\text{[CLS]}}$:
$$\hat{y} = \text{softmax}(W_{\text{classifier}} \cdot h_{\text{[CLS]}} + b_{\text{classifier}})$$

---

## 4. Key Empirical Results (Held-Out Test Set: 518 Sentences)

| Metric | Baseline (TF-IDF + LogReg) | FinBERT (Domain Transformer) | Delta / Factor |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 85.33% | **97.10%** | **+11.77%** |
| **Macro F1** | 0.7943 | **0.9664** | **+0.1721** |
| **Weighted F1** | 0.8456 | **0.9713** | **+0.1257** |
| **Negative F1 (Bearish)** | 0.7129 | **0.9612** | **+0.2483** |
| **Negative Recall** | 57.14% (Misses 43%) | **98.41%** (Catches 62/63) | **+41.27%** |
| **Positive F1 (Bullish)** | 0.7654 | **0.9603** | **+0.1949** |
| **Neutral F1 (Disclosure)** | 0.9046 | **0.9778** | **+0.0732** |
| **Inference Latency** | **0.21 ms / sample** | 5.00 ms / sample | ~23.8x faster |
| **Throughput** | **4,659 samples / sec** | 200 samples / sec | 23.3x higher |
| **Parameter Footprint** | **30,003 parameters (0.6 MB)** | 109,484,547 parameters (438 MB)| 3,649x smaller |
| **McNemar $\chi^2$** | - | **39.5604** ($p = 0.000000$) | Odds ratio: **5.07** |

---

## 5. Quickstart & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data (Financial PhraseBank)
```bash
python src/data_loader.py
```

### 3. Train Classical Baseline
```bash
python src/train_baseline.py
```

### 4. Run Comprehensive Evaluation
```bash
python src/evaluate.py
```

### 5. Run Test Suite
```bash
pytest tests/test_inference.py -v
```

### 6. Launch Streamlit Application
```bash
streamlit run app.py
```

