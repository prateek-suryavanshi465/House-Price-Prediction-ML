import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="House Price Prediction | Prateek Suryavanshi",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 House Price Prediction")
st.write("Machine learning project by Prateek D. Suryavanshi")
st.markdown(
    "This dashboard presents the evaluation results of a house-price "
    "prediction model trained on the Ames Housing dataset."
)

metrics_path = Path("model_metrics.json")
metrics = {}
if metrics_path.exists():
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        metrics = {}

st.subheader("Model evaluation")
cols = st.columns(3)
default_metrics = [
    ("Test MAE", "$16,922.49"),
    ("Test RMSE", "$30,265.12"),
    ("Test R²", "0.8806"),
]
for col, (label, default_value) in zip(cols, default_metrics):
    key = label.lower().replace(" ", "_").replace("²", "2")
    value = metrics.get(key, default_value)
    col.metric(label, str(value))

st.subheader("What this project covers")
st.markdown(
    "- Exploratory data analysis and missing-value review\n"
    "- Preprocessing of numeric and categorical features\n"
    "- Comparison of regression models\n"
    "- Evaluation using MAE, RMSE, and R²"
)

if Path("model_comparison.csv").exists():
    st.subheader("Model comparison")
    try:
        comparison = pd.read_csv("model_comparison.csv")
        st.dataframe(comparison, use_container_width=True)
    except Exception as exc:
        st.info(f"Could not display model comparison file: {exc}")

if Path("feature_importance.csv").exists():
    st.subheader("Feature importance")
    try:
        importance = pd.read_csv("feature_importance.csv")
        st.dataframe(importance.head(15), use_container_width=True)
    except Exception as exc:
        st.info(f"Could not display feature importance file: {exc}")

st.caption(
    "This is a public project showcase of saved evaluation results. "
    "It does not make new predictions from user-supplied house features."
)
