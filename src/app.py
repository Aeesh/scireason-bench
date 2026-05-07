import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import seaborn as sns

st.set_page_config(
    page_title="SciReason-Bench",
    page_icon="🔬",
    layout="wide"
)

ANALYSIS_DIR = "results/analysis"
SCORED_DIR = "results/scored"

QUESTION_TYPES = ["factual", "conceptual", "numerical", "synthesis", "calibration"]
TYPE_LABELS = {
    "factual": "Factual Recall",
    "conceptual": "Conceptual Explanation",
    "numerical": "Numerical Reasoning",
    "synthesis": "Cross-Domain Synthesis",
    "calibration": "Calibration & Uncertainty"
}

####### Sidebar #######
with st.sidebar:
    st.title("🔬 SciReason-Bench")
    st.markdown(
        "A structured benchmark evaluating **4 LLMs** across **5 question types** "
        "covering AI/ML and scientific reasoning."
    )
    st.markdown("---")
    st.markdown("**Models evaluated**")
    st.markdown("- Llama 3.2 3B (Meta, local)")
    st.markdown("- Mistral 7B (Mistral AI, local)")
    st.markdown("- Phi-3 Mini (Microsoft, local)")
    st.markdown("- Gemini 1.5 Flash (Google, cloud)")
    st.markdown("---")
    st.markdown("**Benchmark stats**")
    st.markdown("- 100 questions")
    st.markdown("- 5 question types, 20 each")
    st.markdown("- Scored 0-3 by Gemini Flash judge")
    st.markdown("- Domains: AI/ML, Materials Science, Cross-domain")

####### Load data #######
@st.cache_data
def load_data():
    df = pd.read_csv(f"{ANALYSIS_DIR}/all_results.csv")
    summary = pd.read_csv(f"{ANALYSIS_DIR}/overall_summary.csv", index_col=0)
    type_summary = pd.read_csv(f"{ANALYSIS_DIR}/type_summary.csv", index_col=0)
    diff_summary = pd.read_csv(f"{ANALYSIS_DIR}/difficulty_summary.csv", index_col=0)
    return df, summary, type_summary, diff_summary

df, summary, type_summary, diff_summary = load_data()

####### Header #######
st.title("SciReason-Bench: LLM Evaluation on Scientific Reasoning")
st.markdown(
    "**Research question:** How do small open-source LLMs compare to a commercial model "
    "across different scientific reasoning tasks, and does model size predict performance "
    "consistently?"
)

####### Key findings #######
st.markdown("---")
st.markdown("### Key Findings")
col1, col2, col3, col4 = st.columns(4)

models_sorted = summary.sort_values("Mean Score %", ascending=False)
for col, (model, row) in zip([col1, col2, col3, col4], models_sorted.iterrows()):
    with col:
        st.metric(label=model, value=f"{row['Mean Score %']:.1f}%", delta=None)

st.markdown("---")

####### Tab layout #######
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🗂️ By Question Type",
    "📈 By Difficulty",
    "🔍 Question Explorer",
    "⚖️ Head-to-Head"
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Overall Scores")
        st.image(f"{ANALYSIS_DIR}/overall_scores.png")
    with col2:
        st.markdown("#### Capability Profiles")
        st.image(f"{ANALYSIS_DIR}/radar_chart.png")

    st.markdown("#### Average Latency")
    if os.path.exists(f"{ANALYSIS_DIR}/latency.png"):
        st.image(f"{ANALYSIS_DIR}/latency.png")

with tab2:
    st.markdown("#### Performance by Question Type")
    st.image(f"{ANALYSIS_DIR}/type_heatmap.png")
    st.markdown("---")
    st.markdown("#### Per-Type Scores Table")
    display_cols = {t: TYPE_LABELS[t] for t in QUESTION_TYPES if t in type_summary.columns}
    st.dataframe(type_summary.rename(columns=display_cols).style.background_gradient(
        cmap="RdYlGn", vmin=0, vmax=100
    ))

with tab3:
    st.markdown("#### Performance by Difficulty")
    st.image(f"{ANALYSIS_DIR}/difficulty_scores.png")
    st.dataframe(diff_summary.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=100))

with tab4:
    st.markdown("#### Explore Individual Questions")

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_type = st.selectbox("Question type:", ["All"] + QUESTION_TYPES)
    with col2:
        selected_difficulty = st.selectbox("Difficulty:", ["All", "easy", "medium", "hard"])
    with col3:
        selected_model = st.selectbox("Model:", ["All"] + df["model"].unique().tolist())

    filtered = df.copy()
    if selected_type != "All":
        filtered = filtered[filtered["type"] == selected_type]
    if selected_difficulty != "All":
        filtered = filtered[filtered["difficulty"] == selected_difficulty]
    if selected_model != "All":
        filtered = filtered[filtered["model"] == selected_model]

    st.dataframe(filtered[[
        "question_id", "model", "type", "difficulty", "score_pct"
    ]].rename(columns={
        "question_id": "ID",
        "model": "Model",
        "type": "Type",
        "difficulty": "Difficulty",
        "score_pct": "Score %"
    }))

    # Full question viewer
    st.markdown("---")
    selected_qid = st.selectbox(
        "Select a question to inspect:",
        df["question_id"].unique().tolist()
    )
    q_rows = df[df["question_id"] == selected_qid]
    if not q_rows.empty:
        st.markdown(f"**Question ID:** {selected_qid}")
        st.markdown(f"**Type:** {q_rows.iloc[0]['type']} | **Difficulty:** {q_rows.iloc[0]['difficulty']}")

        # Load full responses from scored files
        for model_key in df["model_key"].unique():
            scored_file = f"{SCORED_DIR}/{model_key}_scored.json"
            if os.path.exists(scored_file):
                with open(scored_file) as f:
                    scored = json.load(f)
                for item in scored["responses"]:
                    if item["id"] == selected_qid:
                        with st.expander(
                            f"{scored['model_info']['display_name']} — Score: {item['score']}/3"
                        ):
                            st.markdown(f"**Response:** {item['model_response']}")
                            st.markdown(f"**Judge reasoning:** {item.get('score_reasoning', 'N/A')}")

with tab5:
    st.markdown("#### Head-to-Head Comparison")
    col1, col2 = st.columns(2)
    with col1:
        model_a = st.selectbox("Model A:", df["model"].unique().tolist(), key="a")
    with col2:
        model_b = st.selectbox(
            "Model B:",
            [m for m in df["model"].unique().tolist() if m != model_a],
            key="b"
        )

    a_scores = df[df["model"] == model_a].groupby("type")["score_pct"].mean()
    b_scores = df[df["model"] == model_b].groupby("type")["score_pct"].mean()

    comparison = pd.DataFrame({
        model_a: a_scores,
        model_b: b_scores
    }).fillna(0)
    comparison["Difference (A-B)"] = comparison[model_a] - comparison[model_b]
    comparison.index = [TYPE_LABELS.get(t, t) for t in comparison.index]

    st.dataframe(comparison.style.background_gradient(
        subset=["Difference (A-B)"], cmap="RdYlGn", vmin=-30, vmax=30
    ).format("{:.1f}"))

st.markdown("---")
st.caption(
    "SciReason-Bench · 100 questions · 4 models · Scored by Gemini 1.5 Flash judge · "
    "Question types: Factual, Conceptual, Numerical, Synthesis, Calibration"
)