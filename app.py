"""
Financial Sentiment Classification Streamlit Application.
Comparative Evaluation: TF-IDF + Logistic Regression vs. ProsusAI/finbert.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Financial Sentiment Classification | Baseline vs FinBERT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 0.5rem;
    }
    .sentiment-positive {
        color: #16A34A;
        font-weight: 700;
        background: #DCFCE7;
        padding: 4px 10px;
        border-radius: 6px;
    }
    .sentiment-negative {
        color: #DC2626;
        font-weight: 700;
        background: #FEE2E2;
        padding: 4px 10px;
        border-radius: 6px;
    }
    .sentiment-neutral {
        color: #475569;
        font-weight: 700;
        background: #F1F5F9;
        padding: 4px 10px;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_unified_predictor():
    """Initializes and caches the unified predictor with both models."""
    from src.predict import UnifiedPredictor
    return UnifiedPredictor(config_path="config/config.yaml")


@st.cache_data
def load_metrics():
    metrics_path = Path("results/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_error_analysis():
    err_path = Path("results/error_analysis.json")
    if err_path.exists():
        with open(err_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select View",
    [
        "🏠 Overview & Theory",
        "🔍 Live Inference Playground",
        "📊 Benchmark & Metrics",
        "🔬 Linguistic & Error Analysis",
        "📁 Batch CSV Processing",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Specifications:**")
st.sidebar.markdown("- **Task:** 3-Class Sentiment Classification")
st.sidebar.markdown("- **Classes:** Negative (0), Neutral (1), Positive (2)")
st.sidebar.markdown("- **Dataset:** Financial PhraseBank (Malo et al.)")
st.sidebar.markdown("- **Baseline:** TF-IDF (1-2) + Logistic Regression")
st.sidebar.markdown("- **Transformer:** ProsusAI/finbert (BERT-base)")


# Helper to render sentiment badge
def render_badge(label_name: str) -> str:
    lbl = label_name.lower()
    if lbl == "positive":
        return f'<span class="sentiment-positive">POSITIVE</span>'
    elif lbl == "negative":
        return f'<span class="sentiment-negative">NEGATIVE</span>'
    return f'<span class="sentiment-neutral">NEUTRAL</span>'


# =========================================================================
# Page 1: Overview & Mathematical Foundations
# =========================================================================
if page == "🏠 Overview & Theory":
    st.markdown('<div class="main-title">Sentiment Classification of Financial News and Disclosures</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A Rigorous Comparative Study: Classical TF-IDF + Logistic Regression vs. ProsusAI/FinBERT</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("1. Project Overview & Core Research Question")
        st.markdown(
            """
            Financial sentiment classification presents unique challenges not encountered in generic sentiment domains (such as movie or product reviews):
            * **Domain Terminology:** Words like *"liability"*, *"depreciation"*, *"margin compression"*, or *"amortization"* denote standard business operations rather than overt emotional negativity.
            * **Polarity in Numbers and Symbols:** Punctuation, currency signs (`$`, `€`), and percentages (`+14%` vs `-14%`) dictate polarity.
            * **Information Asymmetry:** Up to **62%** of financial texts are purely objective regulatory disclosures (*Neutral*), making class imbalance and subtle semantic deviations critical.
            
            **The Core Research Question:**
            > *Why do general bag-of-words models fail on financial syntax, and how much measurable accuracy does contextual domain pre-training add relative to inference latency and parameter footprint?*
            """
        )

    with col2:
        st.subheader("Key Architecture Comparison")
        st.markdown(
            """
            | Property | Baseline (TF-IDF + LR) | FinBERT (Domain Transformer) |
            | :--- | :--- | :--- |
            | **Architecture** | Unigrams + Bigrams + LogReg | 12-layer Bidirectional Transformer |
            | **Pre-training** | None (Trained on split) | Reuters TRC2 Financial Corpus |
            | **Parameters** | ~30,000 coefficients | ~109.5 Million parameters |
            | **Context Sensitivity** | Bag-of-Words (No order) | Self-Attention (Order & Syntax aware) |
            | **Avg. Test Latency** | **0.21 ms / sample** | **5.00 ms / sample** (~24x slower) |
            | **Macro F1 Score** | **0.7943** | **0.9664** (+17.2% absolute F1 gain) |
            """
        )

    st.markdown("---")
    st.subheader("2. Mathematical Foundations")

    tab_a, tab_b, tab_c = st.tabs([
        "A. TF-IDF Representation (Sublinear Scaling)",
        "B. Multinomial Logistic Regression",
        "C. Scaled Dot-Product Attention in FinBERT",
    ])

    with tab_a:
        st.markdown("#### Sublinear Term Frequency - Inverse Document Frequency")
        st.markdown(
            r"""
            To mitigate dominance by frequently repeated tokens in earnings reports, sublinear term frequency scaling is applied:
            $$\text{TF}(t, d) = 1 + \log(\text{count}(t, d)) \quad \text{if } \text{count}(t, d) > 0, \quad \text{else } 0$$
            
            Smooth inverse document frequency prevents zero-division for rare out-of-vocabulary terms across $N$ corpus documents:
            $$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$
            
            The combined token weight is computed as:
            $$w(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$
            
            Each document vector is subsequently $L_2$-normalized:
            $$v_{\text{norm}} = \frac{v}{\Vert{}v\Vert{}_2} = \frac{v}{\sqrt{\sum_{i=1}^d v_i^2}}$$
            """
        )

    with tab_b:
        st.markdown("#### Multinomial Logistic Regression ($K=3$)")
        st.markdown(
            r"""
            For $K = 3$ mutually exclusive sentiment classes with weight matrix $W \in \mathbb{R}^{K \times d}$ and bias vector $b \in \mathbb{R}^K$:
            $$P(y = k \mid x) = \frac{\exp(w_k^T x + b_k)}{\sum_{j=1}^K \exp(w_j^T x + b_j)}$$
            
            Optimal weights are obtained by minimizing the cross-entropy loss with $L_2$ regularisation parameter $C$:
            $$\mathcal{L}(W) = \frac{1}{2}\Vert{}W\Vert{}_F^2 + C \sum_{i=1}^M \sum_{k=1}^K -y_{ik} \log(P(y_i = k \mid x_i))$$
            
            Where $C = \frac{1}{\lambda}$ controls the regularisation strength (tuned via validation grid search).
            """
        )

    with tab_c:
        st.markdown("#### Scaled Dot-Product Attention and FinBERT Architecture")
        st.markdown(
            r"""
            Given Query ($Q$), Key ($K$), and Value ($V$) projections with attention head dimension $d_k = 64$:
            $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
            
            Multi-head attention projects across $h = 12$ independent representation subspaces:
            $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_{12})W^O$$
            
            Final 3-class sentiment prediction is computed from the pooled $[CLS]$ token embedding $h_{\text{[CLS]}}$:
            $$\hat{y} = \text{softmax}(W_{\text{classifier}} \cdot h_{\text{[CLS]}} + b_{\text{classifier}})$$
            """
        )


# =========================================================================
# Page 2: Live Inference Playground
# =========================================================================
elif page == "🔍 Live Inference Playground":
    st.markdown('<div class="main-title">Live Inference Playground</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Test single or multi-sentence financial statements across both models in real time.</div>', unsafe_allow_html=True)

    predictor = get_unified_predictor()

    preset_options = {
        "Custom Input": "",
        "Bullish Earnings (Clear Positive)": "Operating profit surged by 28% YoY to EUR 145m, beating consensus analyst forecasts.",
        "Margin Compression Trap (Contextual Negative)": "Net loss widened significantly due to severe margin compression and raw material inflation.",
        "Corporate Governance (Objective Neutral)": "The board of directors announced that the Annual General Meeting will convene on April 28.",
        "Syntactic Negation (Tricky Negative)": "The restructuring program did not prevent a sharp drop in regional quarterly revenues.",
        "Financial Distress (Missing Emotional Words)": "Company filed for bankruptcy reorganization following an unsustainable debt default.",
        "Cautious Outlook (Subtle Bearish)": "Management anticipates full-year sales to remain flat or modestly contract amidst macroeconomic headwind.",
    }

    selected_preset = st.selectbox("Select a benchmark financial headline or enter custom text:", list(preset_options.keys()))
    default_text = preset_options[selected_preset]

    input_text = st.text_area(
        "Financial Headline / Disclosure Text:",
        value=default_text,
        height=100,
        placeholder="Type or paste any financial sentence here (e.g. 'Operating profit rose 15% to $400M')...",
    )

    col_btn, _ = st.columns([1, 4])
    run_btn = col_btn.button("Analyze Sentiment", type="primary", use_container_width=True)

    if run_btn or input_text:
        if not input_text.strip():
            st.warning("Please enter a sentence to evaluate.")
        else:
            with st.spinner("Executing dual-model inference..."):
                results = predictor.predict_both(input_text)[0]
                interpretability = predictor.get_baseline_feature_contributions(input_text)

            b_info = results["baseline"]
            f_info = results["finbert"]

            st.markdown("### Model Predictions Comparison")
            col_b, col_f = st.columns(2)

            with col_b:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4>Model 1: TF-IDF + Logistic Regression</h4>
                        <p><strong>Predicted Class:</strong> {render_badge(b_info['label_name'])}</p>
                        <p><strong>Confidence:</strong> {b_info['confidence']*100:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("**Probability Distribution:**")
                st.progress(b_info["probabilities"]["negative"], text=f"Negative: {b_info['probabilities']['negative']*100:.1f}%")
                st.progress(b_info["probabilities"]["neutral"], text=f"Neutral: {b_info['probabilities']['neutral']*100:.1f}%")
                st.progress(b_info["probabilities"]["positive"], text=f"Positive: {b_info['probabilities']['positive']*100:.1f}%")

            with col_f:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4>Model 2: ProsusAI/finbert (Domain Transformer)</h4>
                        <p><strong>Predicted Class:</strong> {render_badge(f_info['label_name'])}</p>
                        <p><strong>Confidence:</strong> {f_info['confidence']*100:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("**Probability Distribution:**")
                st.progress(f_info["probabilities"]["negative"], text=f"Negative: {f_info['probabilities']['negative']*100:.1f}%")
                st.progress(f_info["probabilities"]["neutral"], text=f"Neutral: {f_info['probabilities']['neutral']*100:.1f}%")
                st.progress(f_info["probabilities"]["positive"], text=f"Positive: {f_info['probabilities']['positive']*100:.1f}%")

            # Agreement Status
            st.markdown("---")
            if results["agreement"]:
                st.success("✅ **Consensus:** Both models agreed on the predicted sentiment class.")
            else:
                st.warning(
                    f"⚠️ **Model Disagreement:** Baseline predicted **{b_info['label_name'].upper()}** "
                    f"while FinBERT predicted **{f_info['label_name'].upper()}**. "
                    "This highlights the divergence between bag-of-words token weights and full contextual self-attention."
                )

            # Feature Interpretability for Baseline
            with st.expander("🔬 TF-IDF Baseline Feature Interpretability & Token Attribution"):
                st.write(f"**Cleaned Tokens Detected:** `{', '.join(interpretability.get('active_tokens', []))}`")
                contrib_cols = st.columns(3)
                for i, class_name in enumerate(["negative", "neutral", "positive"]):
                    with contrib_cols[i]:
                        st.write(f"**Top positive weights for {class_name.capitalize()}:**")
                        items = interpretability.get("class_contributions", {}).get(class_name, [])
                        if items:
                            for item in items:
                                st.markdown(f"- `{item['token']}`: `+{item['score']:.4f}`")
                        else:
                            st.caption("No significant active tokens.")


# =========================================================================
# Page 3: Benchmark & Metrics Dashboard
# =========================================================================
elif page == "📊 Benchmark & Metrics":
    st.markdown('<div class="main-title">Evaluation Benchmark & Statistical Testing</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluated on the held-out 15% stratified test split (518 financial sentences).</div>', unsafe_allow_html=True)

    metrics = load_metrics()
    if not metrics:
        st.error("Metrics file results/metrics.json not found. Run `python -m src.evaluate` first.")
    else:
        b_met = metrics["baseline_model"]["metrics"]
        f_met = metrics["finbert_model"]["metrics"]

        # Top KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("FinBERT Accuracy", f"{f_met['accuracy']*100:.2f}%", f"+{(f_met['accuracy']-b_met['accuracy'])*100:.2f}% vs Baseline")
        kpi2.metric("FinBERT Macro-F1", f"{f_met['macro_avg']['f1']:.4f}", f"+{f_met['macro_avg']['f1']-b_met['macro_avg']['f1']:.4f} vs Baseline")
        kpi3.metric("Baseline Latency", f"{metrics['baseline_model']['latency']['ms_per_sample']:.2f} ms", "23x faster than FinBERT", delta_color="inverse")
        kpi4.metric("FinBERT Latency", f"{metrics['finbert_model']['latency']['ms_per_sample']:.2f} ms", "200 samples / sec (GPU)")

        st.markdown("---")

        # Tabbed Metrics Breakdown
        tab_comp, tab_cm, tab_mcnemar = st.tabs([
            "📋 Side-by-Side Performance Comparison",
            "🖼️ Confusion Matrices",
            "📐 McNemar's Statistical Significance Test",
        ])

        with tab_comp:
            st.subheader("Classification Performance Breakdown")
            comp_data = {
                "Metric": [
                    "Overall Accuracy",
                    "Macro Precision",
                    "Macro Recall",
                    "Macro F1 Score",
                    "Weighted F1 Score",
                    "Negative Class F1 (Bearish)",
                    "Negative Class Recall",
                    "Neutral Class F1 (Disclosures)",
                    "Positive Class F1 (Bullish)",
                    "Positive Class Recall",
                    "Parameter Footprint",
                    "Inference Latency / Sample",
                    "Throughput (Samples / sec)",
                ],
                "Baseline (TF-IDF + LR)": [
                    f"{b_met['accuracy']:.4f}",
                    f"{b_met['macro_avg']['precision']:.4f}",
                    f"{b_met['macro_avg']['recall']:.4f}",
                    f"{b_met['macro_avg']['f1']:.4f}",
                    f"{b_met['weighted_avg']['f1']:.4f}",
                    f"{b_met['per_class']['negative']['f1']:.4f}",
                    f"{b_met['per_class']['negative']['recall']:.4f} (Misses 43% of Bearish)",
                    f"{b_met['per_class']['neutral']['f1']:.4f}",
                    f"{b_met['per_class']['positive']['f1']:.4f}",
                    f"{b_met['per_class']['positive']['recall']:.4f}",
                    f"{metrics['baseline_model']['footprint']['parameter_count']:,} params ({metrics['baseline_model']['footprint']['disk_size_mb']} MB)",
                    f"{metrics['baseline_model']['latency']['ms_per_sample']:.2f} ms",
                    f"{metrics['baseline_model']['latency']['throughput_samples_per_sec']:.1f}",
                ],
                "FinBERT (Transformer)": [
                    f"{f_met['accuracy']:.4f}",
                    f"{f_met['macro_avg']['precision']:.4f}",
                    f"{f_met['macro_avg']['recall']:.4f}",
                    f"{f_met['macro_avg']['f1']:.4f}",
                    f"{f_met['weighted_avg']['f1']:.4f}",
                    f"{f_met['per_class']['negative']['f1']:.4f}",
                    f"{f_met['per_class']['negative']['recall']:.4f} (Captures 98.4%)",
                    f"{f_met['per_class']['neutral']['f1']:.4f}",
                    f"{f_met['per_class']['positive']['f1']:.4f}",
                    f"{f_met['per_class']['positive']['recall']:.4f} (Captures 100%)",
                    f"{metrics['finbert_model']['footprint']['total_parameters']:,} params (~438 MB)",
                    f"{metrics['finbert_model']['latency']['ms_per_sample']:.2f} ms",
                    f"{metrics['finbert_model']['latency']['throughput_samples_per_sec']:.1f}",
                ],
                "Delta (FinBERT - Baseline)": [
                    f"+{(f_met['accuracy'] - b_met['accuracy'])*100:.2f}%",
                    f"+{(f_met['macro_avg']['precision'] - b_met['macro_avg']['precision'])*100:.2f}%",
                    f"+{(f_met['macro_avg']['recall'] - b_met['macro_avg']['recall'])*100:.2f}%",
                    f"+{(f_met['macro_avg']['f1'] - b_met['macro_avg']['f1'])*100:.2f}%",
                    f"+{(f_met['weighted_avg']['f1'] - b_met['weighted_avg']['f1'])*100:.2f}%",
                    f"+{(f_met['per_class']['negative']['f1'] - b_met['per_class']['negative']['f1'])*100:.2f}%",
                    f"+{(f_met['per_class']['negative']['recall'] - b_met['per_class']['negative']['recall'])*100:.2f}%",
                    f"+{(f_met['per_class']['neutral']['f1'] - b_met['per_class']['neutral']['f1'])*100:.2f}%",
                    f"+{(f_met['per_class']['positive']['f1'] - b_met['per_class']['positive']['f1'])*100:.2f}%",
                    f"+{(f_met['per_class']['positive']['recall'] - b_met['per_class']['positive']['recall'])*100:.2f}%",
                    "3,649x larger footprint",
                    "+4.78 ms overhead",
                    "23x lower throughput",
                ],
            }
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

        with tab_cm:
            st.subheader("Dual Normalized Confusion Matrices")
            cm_img_path = Path("results/confusion_matrices.png")
            if cm_img_path.exists():
                st.image(str(cm_img_path), caption="Normalized Confusion Matrices (Baseline vs. FinBERT on Held-Out Test Set)")
            else:
                st.info("Run evaluate.py to generate confusion matrices plot.")

        with tab_mcnemar:
            st.subheader("McNemar's Statistical Significance Test")
            mcn = metrics["mcnemar_significance_test"]
            st.markdown(
                f"""
                McNemar's test assesses whether the disagreement rates between the two classifiers
                differ significantly on the paired 518 held-out test predictions:
                
                * **Test Statistic ($\\chi^2$):** `{mcn['mcnemar_chi2']}`
                * **p-value:** `{mcn['p_value']:.6f}`
                * **Odds Ratio:** `{mcn['odds_ratio']}`
                * **Statistical Significance:** `{'Significant (p < 0.05)' if mcn['statistically_significant (alpha=0.05)'] else 'Not Significant'}`
                
                **Contingency Table Breakdown:**
                * $n_{{00}}$ (Both models correct): **{mcn['contingency_table']['both_correct (n00)']}**
                * $n_{{01}}$ (Baseline correct, FinBERT incorrect): **{mcn['contingency_table']['baseline_correct_finbert_incorrect (n01)']}**
                * $n_{{10}}$ (Baseline incorrect, FinBERT correct): **{mcn['contingency_table']['baseline_incorrect_finbert_correct (n10)']}**
                * $n_{{11}}$ (Both models incorrect): **{mcn['contingency_table']['both_incorrect (n11)']}**
                
                > **Conclusion:** *{mcn['interpretation']}* With an odds ratio of **{mcn['odds_ratio']}**, FinBERT is over 5 times more likely to correctly classify a financial statement when the baseline errs than vice versa.
                """
            )


# =========================================================================
# Page 4: Error Analysis & Linguistic Deep Dive
# =========================================================================
elif page == "🔬 Linguistic & Error Analysis":
    st.markdown('<div class="main-title">Linguistic & Error Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Why Bag-of-Words Models Fail on Financial Disclosures</div>', unsafe_allow_html=True)

    err_data = load_error_analysis()
    if not err_data:
        st.warning("Error analysis data not found. Run evaluate.py first.")
    else:
        summary = err_data["summary"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Sentences", summary["total_test_samples"])
        c2.metric("Model Agreements", f"{summary['agreements_count']} ({summary['agreement_rate']*100:.1f}%)")
        c3.metric("Disagreements", summary["disagreements_count"])
        c4.metric("FinBERT Superior Cases", summary["finbert_only_correct"])

        st.markdown("---")
        st.subheader("Disagreement Sample Explorer")

        category_filter = st.selectbox(
            "Filter by Disagreement Category:",
            [
                "All Disagreements",
                "FinBERT Correct, Baseline Failed (Contextual Financial Nuance)",
                "Baseline Correct, FinBERT Failed",
            ],
        )

        disagreements = err_data["disagreements"]
        if category_filter != "All Disagreements":
            disagreements = [d for d in disagreements if d.get("disagreement_category") == category_filter]

        st.write(f"Showing **{len(disagreements)}** disagreement instances:")

        table_rows = []
        for d in disagreements:
            table_rows.append({
                "Index": d["index"],
                "Financial Sentence": d["sentence"],
                "Ground Truth": d["true_label_name"].upper(),
                "Baseline Pred": d["baseline"]["pred_label_name"].upper(),
                "Baseline Conf": f"{d['baseline']['confidence']*100:.1f}%",
                "FinBERT Pred": d["finbert"]["pred_label_name"].upper(),
                "FinBERT Conf": f"{d['finbert']['confidence']*100:.1f}%",
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        st.markdown("---")
        st.subheader("Linguistic Reasons for Bag-of-Words Breakdown")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(
                r"""
                #### 1. Severe False-Neutral Bias on Bearish News
                In our empirical results, the baseline achieved only **57.1% Recall** on negative financial news.
                Because financial journalists avoid dramatic adjectives (*"terrible"*, *"catastrophic"*) in favor of measured institutional prose:
                * *"Operating profit fell to EUR 4.2m from EUR 11.5m"*
                * *"The company recorded an impairment on goodwill."*
                
                The baseline sees unigrams like *"operating"*, *"profit"*, *"EUR"*, and assigns higher probability to the majority class (**Neutral** or **Positive**).
                """
            )
        with col_r2:
            st.markdown(
                r"""
                #### 2. Syntactic Negation and Multi-Token Modifiers
                Linear models compute $\sum w_i x_i$. Even with bigrams:
                * *"did not avoid bankruptcy"* has unigrams *"not"*, *"avoid"*, *"bankruptcy"*.
                * *"decreased less than expected"* contains *"decreased"* (bearish token) but represents bullish relative performance.
                
                FinBERT's multi-head self-attention attends bidirectionally across the full sentence, routing the effect of *"not"* or *"exceeded"* directly into the verb and metric head.
                """
            )


# =========================================================================
# Page 5: Batch CSV Processing
# =========================================================================
elif page == "📁 Batch CSV Processing":
    st.markdown('<div class="main-title">Batch Financial CSV Inference</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a CSV of financial headlines, earnings disclosures, or regulatory filings to run parallel inference.</div>', unsafe_allow_html=True)

    predictor = get_unified_predictor()

    uploaded_file = st.file_uploader("Upload CSV file (must contain a text/sentence column):", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write(f"Uploaded CSV has **{len(df_uploaded)}** rows and columns: `{list(df_uploaded.columns)}`")

        text_col = st.selectbox("Select column containing financial text:", list(df_uploaded.columns))

        if st.button("Process Batch Predictions", type="primary"):
            sentences = df_uploaded[text_col].dropna().astype(str).tolist()
            with st.spinner(f"Evaluating {len(sentences)} sentences with both models..."):
                results = predictor.predict_both(sentences)

            augmented_rows = []
            for r in results:
                augmented_rows.append({
                    "sentence": r["sentence"],
                    "baseline_prediction": r["baseline"]["label_name"],
                    "baseline_confidence": r["baseline"]["confidence"],
                    "finbert_prediction": r["finbert"]["label_name"],
                    "finbert_confidence": r["finbert"]["confidence"],
                    "models_agree": r["agreement"],
                })

            res_df = pd.DataFrame(augmented_rows)
            st.success(f"Successfully processed {len(res_df)} sentences!")
            st.dataframe(res_df, use_container_width=True)

            csv_bytes = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_bytes,
                file_name="financial_sentiment_predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("Upload a CSV file to begin batch processing. Alternatively, test the demo test set:")
        if st.button("Load First 20 Samples from Test Split"):
            test_df = pd.read_csv("data/test.csv").head(20)
            sentences = test_df["sentence"].tolist()
            with st.spinner("Processing test split sample..."):
                results = predictor.predict_both(sentences)

            sample_rows = []
            for i, r in enumerate(results):
                true_lbl = test_df.iloc[i]["label_text"]
                sample_rows.append({
                    "Sentence": r["sentence"],
                    "Ground Truth": true_lbl.upper(),
                    "Baseline Pred": r["baseline"]["label_name"].upper(),
                    "FinBERT Pred": r["finbert"]["label_name"].upper(),
                    "Agreement": "✅" if r["agreement"] else "❌",
                })
            st.dataframe(pd.DataFrame(sample_rows), use_container_width=True)
