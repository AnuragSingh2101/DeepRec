"""Streamlit Dashboard for DeepRec: Two-Stage Deep Learning Video Recommendation System."""

import sys
from pathlib import Path

# Ensure project root is in sys.path so 'src' can be imported regardless of launch directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import pandas as pd
import numpy as np
import streamlit as st

from src.deeprec.config import load_config
from src.deeprec.service.recommender import RecommendationService


st.set_page_config(
    page_title="DeepRec — Two-Stage Deep Video Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_service():
    """Cache the recommendation service instance."""
    try:
        svc = RecommendationService()
        return svc
    except Exception as e:
        st.error(f"Error loading recommendation service: {e}")
        return None


def main():
    st.sidebar.title("🎬 DeepRec")
    st.sidebar.caption("Two-Stage Deep Recommendation System (YouTube 2016)")

    nav = st.sidebar.radio(
        "Navigation",
        [
            "Overview & Architecture",
            "Personalized Recommendations",
            "Recommendation Inspector",
            "Model Evaluation",
            "Experiment Comparison",
        ],
    )

    svc = get_service()
    cfg = load_config()
    paths = cfg["paths"]

    # Check artifact status
    cand_exists = Path(paths["candidate_model_path"]).exists()
    rank_exists = Path(paths["ranking_model_path"]).exists()
    eval_exists = (Path(paths["evaluation_artifact_dir"]) / "evaluation_summary.json").exists()

    # Sidebar system status
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    if svc and svc.is_loaded:
        st.sidebar.success(f"Catalogue: {len(svc.movie_catalogue):,} items")
        st.sidebar.success(f"Indexed Users: {len(svc.user_map):,}")
    else:
        st.sidebar.warning("Datasets not preprocessed.")

    st.sidebar.write(f"Candidate Gen Model: {'✅ Ready' if cand_exists else '❌ Missing'}")
    st.sidebar.write(f"Ranking Model: {'✅ Ready' if rank_exists else '❌ Missing'}")
    st.sidebar.write(f"Evaluation Metrics: {'✅ Available' if eval_exists else '❌ Missing'}")

    # =========================================================================
    # TAB 1: OVERVIEW & ARCHITECTURE
    # =========================================================================
    if nav == "Overview & Architecture":
        st.title("🎬 DeepRec — Two-Stage Deep Video Recommendation System")
        st.markdown("""
        **Research Foundation:** Inspired by Google's landmark paper:  
        *“Deep Neural Networks for YouTube Recommendations”* (Paul Covington, Jay Adams, Emre Sargin, Google, RecSys 2016).
        """)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Catalogue Movies", f"{len(svc.movie_catalogue):,}" if svc and svc.movie_catalogue else "N/A")
        with col2:
            st.metric("Total Users", f"{len(svc.user_map):,}" if svc and svc.user_map else "N/A")
        with col3:
            st.metric("Candidate Stage", "Sequence MLP + Sampled Softmax")
        with col4:
            st.metric("Ranking Stage", "Deep MLP + Feature Transforms")

        st.markdown("---")
        st.subheader("System Architecture")
        st.markdown("""
        ```mermaid
        graph TD
            A[Full Video Catalogue: 1,682 Items] --> B[Stage 1: Deep Candidate Generation]
            H[User History Sequences] --> B
            C[User Demographics & Context] --> B
            B -->|Top-100 High Recall Candidates| D[Stage 2: Deep Ranking Model]
            F[Continuous Features: Popularity, Recency, Ratings, Genre Affinity, Powers x, x^2, sqrt x] --> D
            D -->|Scored & Reranked| E[Final Personalized Feed: Top-10]
        ```
        """)

        st.subheader("Paper vs. Prototype Principles")
        st.markdown(r"""
        | Concept | Covington et al. (YouTube 2016) | DeepRec Implementation |
        | :--- | :--- | :--- |
        | **Funnel** | Millions $\to$ Hundreds $\to$ Dozens | 1,682 Catalogue $\to$ 100 Candidates $\to$ Top 10 |
        | **History Encoding** | Average pooling of video watch embeddings | Masked average pooling excluding padding token `0` |
        | **Objective (Stage 1)** | Extreme multiclass classification via Sampled Softmax | Sampled Softmax cross-entropy with uniform negative sampling |
        | **Continuous Features** | CDF Quantile normalization + powers ($x, x^2, \sqrt{x}$) | StandardScaler + power transforms ($x, x^2, \sqrt{x}$) fitted on train only |
        | **Objective (Stage 2)** | Weighted Logistic Regression (watch-time weighted) | Binary Cross-Entropy on verified user engagement proxy ($\ge 4.0$ rating) |
        """)


    # =========================================================================
    # TAB 2: PERSONALIZED RECOMMENDATIONS
    # =========================================================================
    elif nav == "Personalized Recommendations":
        st.title("🎯 Personalized Two-Stage Recommendations")

        if not svc or not svc.is_loaded:
            st.warning("Model artifacts or processed data not found. Please run training pipelines.")
            return

        col_input, col_config = st.columns([2, 1])
        with col_input:
            user_options = sorted(list(svc.user_map.keys()))
            selected_user = st.selectbox("Select Existing User ID:", user_options, index=0)

        with col_config:
            cand_k = st.slider("Stage 1 Candidates (K):", min_value=10, max_value=200, value=100, step=10)
            top_n = st.slider("Final Recommendations (N):", min_value=5, max_value=25, value=10, step=5)
            exclude_seen = st.checkbox("Exclude Previously Watched Items", value=True)

        if st.button("Generate Recommendations", type="primary"):
            with st.spinner("Generating two-stage recommendations..."):
                res = svc.get_recommendations(
                    user_id=selected_user,
                    candidate_k=cand_k,
                    top_n=top_n,
                    exclude_seen=exclude_seen,
                )

            st.markdown("---")
            # User History
            st.subheader("User Recent Interaction History")
            hist = res.get("history", [])
            if hist:
                cols = st.columns(len(hist))
                for i, h in enumerate(hist):
                    with cols[i]:
                        st.info(f"**{h['title']}**\n\n*Genres:* {', '.join(h['genres'][:2])}")
            else:
                st.write("No historical interactions available (Cold-Start User).")

            st.markdown("---")
            st.subheader(f"Top {top_n} Final Ranked Recommendations (Filtered from {res['candidate_count']} candidates)")

            recs = res["recommendations"]
            table_data = []
            for r in recs:
                table_data.append({
                    "Rank": r["rank"],
                    "Movie Title": r["title"],
                    "Release Year": r["release_year"],
                    "Genres": ", ".join(r["genres"]),
                    "Candidate Score": f"{r['candidate_score']:.4f}",
                    "Ranking Score": f"{r['ranking_score']:.4f}",
                    "Attribution / Reason": r["explanation"],
                })

            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    # =========================================================================
    # TAB 3: RECOMMENDATION INSPECTOR
    # =========================================================================
    elif nav == "Recommendation Inspector":
        st.title("🔍 Candidate & Feature Attribution Inspector")
        st.markdown("Inspect why a specific candidate movie was scored and ranked for a user.")

        if not svc or not svc.is_loaded:
            st.warning("Please preprocess data and train models first.")
            return

        user_options = sorted(list(svc.user_map.keys()))
        selected_user = st.selectbox("Select User ID to Inspect:", user_options, index=0)

        res = svc.get_recommendations(user_id=selected_user, candidate_k=50, top_n=20)
        rec_titles = [f"{r['rank']}. {r['title']} (ID: {r['movie_id']})" for r in res["recommendations"]]
        selected_rec_str = st.selectbox("Select Recommendation to Inspect:", rec_titles)

        rec_idx = int(selected_rec_str.split(".")[0]) - 1
        rec_item = res["recommendations"][rec_idx]

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Candidate Gen Score (Stage 1)", f"{rec_item['candidate_score']:.4f}")
        with c2:
            st.metric("Final Engagement Score (Stage 2)", f"{rec_item['ranking_score']:.4f}")
        with c3:
            st.metric("Release Year", rec_item["release_year"])

        st.subheader("Feature Values for Selected Candidate")
        u_idx = svc.user_map.get(selected_user, 0)
        m_idx = rec_item["movie_idx"]

        if svc.feature_extractor is not None:
            raw_feats = svc.feature_extractor.extract_raw_features(
                user_idx=u_idx, movie_idx=m_idx, candidate_score=rec_item["candidate_score"]
            )
            feat_names = [
                "Candidate Logit", "Item Popularity", "Item Avg Rating", "Log(Rating Count + 1)",
                "User Mean Rating", "Log(User Count + 1)", "Genre Affinity (Cosine)",
                "Interaction Recency (Log Days)", "Release Year Normalized"
            ]
            feat_df = pd.DataFrame({
                "Feature Name": feat_names,
                "Raw Value": [round(float(v), 4) for v in raw_feats],
            })
            st.dataframe(feat_df, use_container_width=True)
            st.info(f"**Explanation:** {rec_item['explanation']}")

    # =========================================================================
    # TAB 4: MODEL EVALUATION
    # =========================================================================
    elif nav == "Model Evaluation":
        st.title("📊 Offline Evaluation Metrics")

        eval_file = Path(paths["evaluation_artifact_dir"]) / "evaluation_summary.json"
        if not eval_file.exists():
            st.warning("No offline evaluation summary found. Run 'python -m src.deeprec.evaluation.evaluate' to generate real metrics.")
            return

        with open(eval_file) as f:
            summary = json.load(f)

        st.markdown(f"**Evaluated on:** Held-out temporal test split ({summary.get('evaluation_users', 'N/A')} users).")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Catalog Coverage (Stage 1)", f"{summary['candidate_generation'].get('catalog_coverage', 0)*100:.1f}%")
        with c2:
            st.metric("Mean Retrieval Latency", f"{summary['latency_ms'].get('retrieval_mean', 0):.2f} ms")
        with c3:
            st.metric("Mean Ranking Latency", f"{summary['latency_ms'].get('ranking_mean', 0):.2f} ms")

        st.subheader("Metric Comparison Across Cutoffs (K=10, 20, 50)")

        metrics_data = []
        for stage_name, stage_key in [("Candidate Generation Only", "candidate_generation"), ("Full Two-Stage (Candidate + Ranker)", "two_stage_pipeline")]:
            if stage_key in summary:
                s = summary[stage_key]
                metrics_data.append({
                    "Pipeline Stage": stage_name,
                    "Recall@10": s.get("recall@10", "N/A"),
                    "Recall@20": s.get("recall@20", "N/A"),
                    "Recall@50": s.get("recall@50", "N/A"),
                    "Precision@10": s.get("precision@10", "N/A"),
                    "NDCG@10": s.get("ndcg@10", "N/A"),
                    "MAP@10": s.get("map@10", "N/A"),
                })

        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

    # =========================================================================
    # TAB 5: EXPERIMENT COMPARISON
    # =========================================================================
    elif nav == "Experiment Comparison":
        st.title("🔬 Benchmark Experiment Comparison")

        exp_file = Path(paths["evaluation_artifact_dir"]) / "experiment_results.csv"
        if not exp_file.exists():
            exp_file = Path("artifacts/experiment_results.csv")

        if not exp_file.exists():
            st.warning("No experiment results found. Run 'python -m src.deeprec.evaluation.experiments' to execute experiments A-G.")
            return

        df_exp = pd.read_csv(exp_file)
        st.dataframe(df_exp, use_container_width=True)

        st.subheader("Model Performance Comparison (Recall@10 vs Recall@50)")
        chart_df = df_exp.set_index("experiment")[["recall@10", "recall@50"]]
        st.bar_chart(chart_df)

        st.subheader("Ranking Quality (NDCG@10)")
        st.bar_chart(df_exp.set_index("experiment")["ndcg@10"])

        st.subheader("Latency Trade-off")
        st.line_chart(df_exp.set_index("experiment")["mean_latency_ms"])


if __name__ == "__main__":
    main()
