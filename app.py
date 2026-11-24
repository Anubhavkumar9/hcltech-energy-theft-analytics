import streamlit as st
import io
import pandas as pd
import plotly.express as px
from src.preprocessing import load_and_prepare, make_demo_data
from src.models import train_and_evaluate

st.set_page_config(page_title="Energy Theft Analytics", page_icon="⚡", layout="wide")
st.title("⚡ Electricity Theft & Abnormal Consumption Analytics")
st.caption("Decision-support prototype. A flagged pattern is not proof of theft.")
st.sidebar.header("Data and settings")
mode = st.sidebar.radio("Data source", ["Synthetic demonstration data", "Upload SGCC-style CSV"])
alpha = st.sidebar.slider("Hybrid supervised weight (α)", 0.0, 1.0, 0.7, 0.05)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42)
if mode == "Upload SGCC-style CSV":
    uploaded = st.sidebar.file_uploader("Upload labelled CSV", type=["csv"])
    target = st.sidebar.text_input("Binary target column", "FLAG")
    ids = st.sidebar.text_input("ID columns to exclude (comma separated)", "CONS_NO")
else:
    uploaded, target, ids = None, "FLAG", ""
try:
    if mode == "Upload SGCC-style CSV":
        if uploaded is None:
            st.info("Upload a labelled CSV with one consumer per row, a binary target (0/1), and numeric reading columns.")
            st.stop()
        raw, X, y, reading_count = load_and_prepare(uploaded, target,
            [x.strip() for x in ids.split(",") if x.strip()])
        data_label = "Uploaded dataset"
    else:
        raw = make_demo_data()
        # Pass a CSV buffer here so this path also works with older cached
        # preprocessing modules that expect a file-like object.
        demo_csv = io.StringIO(raw.to_csv(index=False))
        _, X, y, reading_count = load_and_prepare(demo_csv, "FLAG", [])
        data_label = "Synthetic demonstration dataset"
    with st.spinner("Training models..."):
        result = train_and_evaluate(X, y, alpha=float(alpha), seed=int(seed))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(X):,}")
    c2.metric("Flagged labels", f"{int(y.sum()):,}")
    c3.metric("Positive-label rate", f"{y.mean()*100:.1f}%")
    c4.metric("Reading columns", reading_count)
    st.warning(f"Data source: {data_label}. Synthetic results are illustrative only.")
    st.subheader("Model comparison")
    metrics = pd.DataFrame(result["metrics"])
    st.dataframe(metrics.style.format({c: "{:.3f}" for c in metrics.columns if c != "Model"}),
        use_container_width=True)
    fig = px.bar(metrics.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]),
        x="Model", y="value", color="variable", barmode="group", title="Held-out test metrics")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Investigation priority (test partition)")
    ranked = pd.DataFrame({"Hybrid risk score": result["hybrid_scores"],
        "Anomaly score": result["anomaly_scores"],
        "Actual label (evaluation only)": result["y_test"].values},
        index=result["X_test"].index).sort_values("Hybrid risk score", ascending=False)
    st.dataframe(ranked.head(50), use_container_width=True)
    st.download_button("Download ranked review candidates",
        ranked.to_csv(index=True).encode("utf-8"), "ranked_review_candidates.csv", "text/csv")
    st.subheader("Hybrid confusion matrix (threshold 0.50)")
    st.dataframe(pd.DataFrame(result["confusion"],
        index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
    st.caption("For research-grade results, use chronological/customer-grouped validation, tune thresholds on validation data, and test independently. A model flag requires human review.")
except Exception as e:
    st.error(f"Could not process the data: {e}")
    st.info("Check the binary target column, numeric consumption columns, and dataset format.")
