# src/stream.py
"""
Streamlit app to upload flow CSVs, score them using a trained IsolationForest,
and interactively set threshold to inspect flagged flows.

Run:
    streamlit run src/stream.py
"""
import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from io import BytesIO

# ----- CONFIG -----
DEFAULT_MODEL_DIR = r"C:\Users\LENOVO\Documents\Projects\Bridge\AI\Anomaly\models\.25"
ROOT = os.path.expanduser("~")
PROJECT_DATA_DIR = os.path.join(ROOT, "Documents", "Projects", "Bridge", "AI", "Anomaly", "data")
SAMPLE_PATH = os.path.join("data", "flows_features.csv")         # relative sample inside repo
DEMO_PATH = os.path.join(PROJECT_DATA_DIR, "demo_flows.csv")    # demo file created by make_demo_data.py
# ------------------

st.set_page_config(page_title="NIDS Demo", layout="wide")
st.title("Prototype NIDS — Demo")
st.markdown(
    "Upload a flow CSV, score with a trained IsolationForest, tune threshold, inspect flagged flows. "
    "Prefer using the demo dataset (created from the feature-engineered file) to show realistic 'new' traffic."
)

# Sidebar: model path, input source
st.sidebar.header("Model & input")
model_dir = st.sidebar.text_input("Model directory", value=DEFAULT_MODEL_DIR)

st.sidebar.markdown("**Choose input CSV**")
uploaded_file = st.sidebar.file_uploader("Upload CSV to score (overrides other choices)", type=["csv"])
use_demo = st.sidebar.checkbox("Use demo dataset (data/demo_flows.csv)", value=True)
use_sample = st.sidebar.checkbox("Use sample from repo (data/flows_features.csv)", value=False)

# Load model & scaler
model_exists = os.path.exists(os.path.join(model_dir, "isoforest.pkl")) and os.path.exists(os.path.join(model_dir, "scaler.pkl"))
if model_exists:
    try:
        clf = joblib.load(os.path.join(model_dir, "isoforest.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        st.sidebar.success("Loaded model & scaler")
    except Exception as e:
        st.sidebar.error(f"Error loading model/scaler: {e}")
        clf = None
        scaler = None
else:
    st.sidebar.warning("Model files not found in folder. Set correct model_dir or train model first.")
    clf = None
    scaler = None

# Load dataframe from chosen source
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.info(f"Loaded uploaded CSV ({len(df)} rows)")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
elif use_demo:
    if os.path.exists(DEMO_PATH):
        df = pd.read_csv(DEMO_PATH)
        st.sidebar.info(f"Loaded demo dataset: {os.path.basename(DEMO_PATH)} ({len(df)} rows)")
    else:
        st.sidebar.error(f"Demo dataset not found at: {DEMO_PATH}")
        st.stop()
elif use_sample:
    if os.path.exists(SAMPLE_PATH):
        df = pd.read_csv(SAMPLE_PATH)
        st.sidebar.info(f"Loaded sample dataset: {SAMPLE_PATH} ({len(df)} rows)")
    else:
        st.sidebar.warning("Sample dataset not found at data/flows_features.csv in repo.")
        st.stop()
else:
    st.info("Upload a CSV or enable 'Use demo' / 'Use sample' to begin.")
    st.stop()

# Ensure model loaded
if clf is None or scaler is None:
    st.error("Model or scaler missing. Please set model_dir to a folder containing isoforest.pkl and scaler.pkl.")
    st.stop()

st.write(f"Loaded {len(df)} rows for scoring")

# ---------------------------
# Prepare numeric columns
# ---------------------------
exclude = {"attack_label", "is_attack"}
# detect numeric columns (pandas dtype)
numeric_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
if not numeric_cols:
    numeric_candidates = ["flow_duration", "total_fwd_packets", "total_backward_packets",
                          "total_length_fwd_packets", "total_length_bwd_packets",
                          "bytes_per_sec", "pkts_per_sec", "fwd_to_bwd_ratio"]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]

if len(numeric_cols) == 0:
    st.error("No numeric features detected. Make sure your CSV has numeric flow features.")
    st.stop()

# Build numeric dataframe (fill NAs)
df_numeric = df[numeric_cols].fillna(0).astype(float).copy()

# ---------- ALIGN FEATURES BEFORE SCALING ----------
trained_feature_names = None
if hasattr(scaler, "feature_names_in_"):
    trained_feature_names = list(scaler.feature_names_in_)
elif hasattr(scaler, "n_features_in_"):
    # we'll know expected count but not names
    trained_feature_names = None

if trained_feature_names:
    # Add missing features (fill with zeros)
    missing = [c for c in trained_feature_names if c not in df_numeric.columns]
    if missing:
        st.warning(f"Missing feature(s) expected by scaler: {missing} — filling with zeros.")
        for c in missing:
            df_numeric[c] = 0.0
    # Drop extras
    extra = [c for c in df_numeric.columns if c not in trained_feature_names]
    if extra:
        st.info(f"Dropping extra columns not seen in training: {extra}")
        df_numeric = df_numeric.drop(columns=extra)
    # Reorder to match training
    df_numeric = df_numeric[trained_feature_names]
    X = df_numeric.values
else:
    # fallback: no feature names saved — use count-based heuristics
    X = df_numeric.values
    expected = getattr(scaler, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        # try removing obvious metadata columns like "__perturbed" or "flow_id"
        if "__perturbed" in df_numeric.columns and X.shape[1] - 1 == expected:
            st.warning("Dropping __perturbed column to match scaler feature count.")
            df_numeric = df_numeric.drop(columns="__perturbed")
            X = df_numeric.values
        elif "flow_id" in df_numeric.columns and X.shape[1] - 1 == expected:
            st.warning("Dropping flow_id column to match scaler feature count.")
            df_numeric = df_numeric.drop(columns="flow_id")
            X = df_numeric.values
        else:
            st.error(f"Feature count mismatch: input has {X.shape[1]} features, scaler expects {expected}.")
            st.stop()
# ---------------------------------------------------

# Safe scaling (now X should match expected features)
try:
    X_scaled = scaler.transform(X)
except Exception as e:
    st.error(f"Error during scaling: {e}")
    st.stop()

# Score: higher score -> more anomalous (we take negative of decision_function)
try:
    scores = -clf.decision_function(X_scaled)
except Exception as e:
    st.error(f"Error during model scoring: {e}")
    st.stop()

# Interactive threshold and output
st.sidebar.header("Threshold & view")
default_thresh = float(np.percentile(scores, 80))
threshold = st.sidebar.slider("Anomaly score threshold (increase to flag more rows)", float(np.min(scores)), float(np.max(scores)), default_thresh)
preds = (scores >= threshold).astype(int)

# Prepare df_out
df_out = df.copy()
df_out["anomaly_score"] = scores
df_out["pred_anomaly"] = preds

# Summary metrics
n_flagged = int(preds.sum())
pct_flagged = n_flagged / len(df_out) * 100
st.metric("Flagged anomalies", f"{n_flagged} / {len(df_out)} ({pct_flagged:.2f}%)")
st.markdown(f"**Threshold:** {threshold:.4f} (higher = more anomalies flagged)")

# Show perturbed status if present
if "__perturbed" in df_out.columns:
    perturbed_count = int(df_out["__perturbed"].sum())
    st.sidebar.write(f"Perturbed rows in dataset: {perturbed_count}")
    show_only_perturbed = st.sidebar.checkbox("Show only perturbed rows", value=False)
else:
    show_only_perturbed = False

# Table to display
cols_to_show = []
if "flow_id" in df_out.columns:
    cols_to_show.append("flow_id")
cols_to_show += ["anomaly_score", "pred_anomaly"]
if "__perturbed" in df_out.columns:
    cols_to_show.append("__perturbed")

# View options
st.sidebar.write("View options")
view_choice = st.sidebar.radio("Rows to display", ["Top anomalies", "All rows", "Only flagged", "Only perturbed" if "__perturbed" in df_out.columns else "Only flagged"])

if view_choice == "Top anomalies":
    display_df = df_out[cols_to_show].sort_values("anomaly_score", ascending=False).reset_index(drop=True).head(200)
elif view_choice == "All rows":
    display_df = df_out[cols_to_show].sort_values("anomaly_score", ascending=False).reset_index(drop=True)
elif view_choice == "Only flagged":
    display_df = df_out[df_out["pred_anomaly"]==1][cols_to_show].sort_values("anomaly_score", ascending=False).reset_index(drop=True)
elif view_choice == "Only perturbed":
    display_df = df_out[df_out.get("__perturbed", 0)==1][cols_to_show].sort_values("anomaly_score", ascending=False).reset_index(drop=True)
else:
    display_df = df_out[cols_to_show].sort_values("anomaly_score", ascending=False).reset_index(drop=True)

st.subheader("Scored flows")
st.dataframe(display_df, use_container_width=True)

# Flagged sample view
st.subheader("Flagged anomalies (sample)")
st.dataframe(df_out[df_out["pred_anomaly"]==1].head(500), use_container_width=True)

# Download scored CSV
buffer = BytesIO()
df_out.to_csv(buffer, index=False)
st.download_button("Download scored CSV", data=buffer.getvalue(), file_name="scored_flows.csv", mime="text/csv")

# Score distribution summary
st.subheader("Anomaly score distribution")
try:
    st.bar_chart(pd.DataFrame({"anomaly_score": df_out["anomaly_score"]}))
except Exception:
    st.write(df_out["anomaly_score"].describe())

# Notes
st.markdown(
    """
    **Notes**
    - The demo dataset (`data/demo_flows.csv`) was sampled from `flows_features.csv` and a portion of rows were perturbed
      to simulate unseen/abnormal flows. Perturbed rows are marked by the `__perturbed` column (if present).
    - The app aligns input features to the model/scaler expected features (fills missing numeric features with zeros,
      drops unexpected extras, and reorders columns to match training).
    - For production scoring, use the `score_flows.py` script to batch-score files with the saved model.
    """
)
