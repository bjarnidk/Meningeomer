import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Meningioma 15y Intervention Risk", layout="wide")

ARTIFACT_PATH = "meningioma_rf_model.joblib"

# -----------------------------
# Load artifact
# -----------------------------
@st.cache_resource
def load_artifact(path):
    return joblib.load(path)

artifact = load_artifact(ARTIFACT_PATH)
model = artifact["calibrated_model"]
feature_names = artifact["feature_names"]

validation_A = artifact.get("validation_A", None)
validation_B = artifact.get("validation_B", None)
validation_AB = artifact.get("validation_AB", None)

# -----------------------------
# Helpers
# -----------------------------
def make_row(age, size_mm, location_value, epilepsi, tryk, focal, calcified, edema, feature_names):
    row = {
        "age": age,
        "tumorsize": size_mm,
        "epilepsi": int(epilepsi),
        "tryksympt": int(tryk),
        "focalsympt": int(focal),
        "calcified": int(calcified),
        "edema": int(edema),
    }
    row_df = pd.DataFrame([row])
    # location dummies
    for c in feature_names:
        if c.startswith("location_"):
            row_df[c] = 0
    chosen_col = f"location_{location_value}"
    if chosen_col in feature_names:
        row_df[chosen_col] = 1
    for c in feature_names:
        if c not in row_df.columns:
            row_df[c] = 0
    return row_df[feature_names]

def lookup_bins(prob, bins):
    """Match prob into calibration bins without boundary gaps."""
    if not bins:
        return None
    for i, b in enumerate(bins):
        if i < len(bins) - 1:
            if prob >= b["p_min"] and prob < b["p_max"]:
                return b
        else:  # last bin inclusive
            if prob >= b["p_min"] and prob <= b["p_max"]:
                return b
    # fallback: closest mean_pred
    diffs = [abs(prob - b["mean_pred"]) for b in bins]
    return bins[int(np.argmin(diffs))]

# -----------------------------
# UI
# -----------------------------
st.title("Meningioma 15-Year Intervention Risk")
st.caption("Random Forest (isotonic calibrated). Shows predictions from Center A training and pooled A+B data.")

col1, col2 = st.columns(2)
with col1:
    age_input = st.number_input("Age (years)", min_value=18, max_value=100, value=65, step=1)
    size_input = st.number_input("Tumor size (mm)", min_value=1, max_value=100, value=30, step=1)
    loc_levels = ["(baseline)"] + artifact["location_levels"]
    sel_loc = st.selectbox("Location", loc_levels, index=0)

with col2:
    epilepsi_in  = st.selectbox("Epilepsi", [0, 1], index=0)
    tryk_in       = st.selectbox("Tryksympt", [0, 1], index=0)
    focal_in      = st.selectbox("Focalsympt", [0, 1], index=0)
    calcified_in  = st.selectbox("Calcified", [0, 1], index=1)
    edema_in      = st.selectbox("Edema", [0, 1], index=0)

# Build row
row_df = make_row(age_input, size_input, sel_loc if sel_loc != "(baseline)" else "BASELINE",
                  epilepsi_in, tryk_in, focal_in, calcified_in, edema_in, feature_names)

# Predict probability
with st.spinner("Predicting..."):
    p = float(model.predict_proba(row_df)[:, 1])  # probability of intervention
    risk_pct = p * 100.0

# -----------------------------
# Get bins and CIs
# -----------------------------
ci_A_txt, ci_AB_txt = "", ""
risk_A, risk_AB = None, None
obs_A, obs_AB = None, None

# --- Center A prediction ---
if validation_A and "reliability_bins" in validation_A:
    bin_A = lookup_bins(p, validation_A["reliability_bins"])
    if bin_A:
        risk_A = risk_pct
        ci_A_txt = f"(95% CI {bin_A['ci_low']*100:.1f}–{bin_A['ci_high']*100:.1f}%)"
        obs_A = f"Observed in A: {bin_A['obs_rate']*100:.1f}% (n={bin_A['n']})"

# --- Pooled A+B prediction ---
if validation_AB and "reliability_bins" in validation_AB:
    bin_AB = lookup_bins(p, validation_AB["reliability_bins"])
    if bin_AB:
        risk_AB = risk_pct
        ci_AB_txt = f"(95% CI {bin_AB['ci_low']*100:.1f}–{bin_AB['ci_high']*100:.1f}%)"
        obs_AB = f"Observed in pooled A+B: {bin_AB['obs_rate']*100:.1f}% (n={bin_AB['n']})"

# -----------------------------
# Display
# -----------------------------
st.subheader("Predicted probability of intervention within 15 years")

col1, col2 = st.columns(2)
with col1:
    if risk_A is not None:
        st.metric("Center A", f"{risk_A:.1f}% {ci_A_txt}")
        if obs_A: st.caption(obs_A)
with col2:
    if risk_AB is not None:
        st.metric("Pooled A+B", f"{risk_AB:.1f}% {ci_AB_txt}")
        if obs_AB: st.caption(obs_AB)

# -----------------------------
# Model card
# -----------------------------
with st.expander("Model card / notes"):
    st.markdown(f"""
- **Model:** {artifact['model_type']}
- **Validation (Center B):** AUC = {validation_B['auc']:.3f}, Brier = {validation_B['brier']:.3f}  
- **Observed reliability:** Predictions shown separately for Center A and pooled A+B.  
- **CI:** Based on Wilson intervals within calibration bins.  
- **Use case:** Estimating risk of intervention for incidental meningioma.  
- **Caution:** Clinical judgment required; small n in some bins → wide CI.
""")
