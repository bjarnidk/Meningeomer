import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.stats.proportion import proportion_confint

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
train_ranges = artifact["train_ranges"]

validation_B = artifact["validation_B"]
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
        else:  # last bin, inclusive
            if prob >= b["p_min"] and prob <= b["p_max"]:
                return b
    # fallback: closest mean_pred
    diffs = [abs(prob - b["mean_pred"]) for b in bins]
    return bins[int(np.argmin(diffs))]

def compute_bin_ci(bin_entry, alpha=0.05):
    """Wilson CI for bin observed rate."""
    if not bin_entry or bin_entry["n"] == 0:
        return None
    count = int(bin_entry["obs_rate"] * bin_entry["n"])
    ci_low, ci_high = proportion_confint(
        count, bin_entry["n"], alpha=alpha, method="wilson"
    )
    return ci_low, ci_high

# -----------------------------
# UI
# -----------------------------
st.title("Meningioma 15-Year Intervention Risk")
st.caption("Random Forest (isotonic calibrated). Trained on Center A, validated on Center B. Risk is probability of intervention within 15 years.")

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

# Predict
with st.spinner("Predicting..."):
    p = float(model.predict_proba(row_df)[:, 1][0])

risk_pct = p * 100.0

# -----------------------------
# CI via pooled bins
# -----------------------------
ci_txt = ""
bin_B, bin_AB = None, None

if "reliability_bins" in validation_B:
    bin_B = lookup_bins(p, validation_B["reliability_bins"])
if validation_AB and "reliability_bins" in validation_AB:
    bin_AB = lookup_bins(p, validation_AB["reliability_bins"])
    if bin_AB:
        ci = compute_bin_ci(bin_AB)
        if ci:
            ci_low, ci_high = ci
            ci_txt = f"(95% CI {ci_low*100:.1f}–{ci_high*100:.1f}%)"

# -----------------------------
# Display
# -----------------------------
st.subheader("Estimated probability of intervention within 15 years")
st.metric(label="Risk", value=f"{risk_pct:.1f}% {ci_txt}")

if bin_B:
    st.caption(
        f"Observed in Center B: {bin_B['obs_rate']*100:.1f}% "
        f"(n={bin_B['n']}) in this risk band."
    )
if bin_AB:
    st.caption(
        f"Observed in pooled A+B: {bin_AB['obs_rate']*100:.1f}% "
        f"(n={bin_AB['n']}) in this risk band."
    )

# -----------------------------
# Model card
# -----------------------------
with st.expander("Model card / notes"):
    st.markdown(f"""
- **Model:** {artifact['model_type']}
- **External validation (B):** AUC = {validation_B['auc']:.3f}, Brier = {validation_B['brier']:.3f}  
- **Observed reliability:** External (B) and pooled (A+B) shown above.  
- **CI:** Based on pooled A+B observed rates (Wilson method).  
- **Use case:** Estimating risk of intervention for incidental meningioma.  
- **Caution:** Clinical judgment required; predictions may be unstable at edges of training distribution.
""")
