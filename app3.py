import os
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Meningioma Risk (15y intervention)", layout="wide")
st.title("Meningioma 15-Year Intervention Risk")
st.caption("Random Forest (isotonic calibrated). Trained on Center A, validated on Center B. Built for rule-out decisions with guardrails.")

# -----------------------------
# Load model artifact
# -----------------------------
@st.cache_resource
def load_artifact(path="meningioma_rf_model.joblib"):
    return joblib.load(path)

artifact = load_artifact()
model = artifact["calibrated_model"]
feature_names = artifact["feature_names"]
train_ranges = artifact["train_ranges"]
valB = artifact.get("validation_B", {})

# -----------------------------
# Predict helper (single row)
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
    for c in feature_names:
        if c.startswith("location_"):
            row_df[c] = 0
    chosen_col = f"location_{location_value}"
    if chosen_col in feature_names:
        row_df[chosen_col] = 1
    for c in feature_names:
        if c not in row_df.columns:
            row_df[c] = 0
    row_df = row_df[feature_names]
    return row_df

# -----------------------------
# UI – Patient sliders / inputs
# -----------------------------
st.header("Interactive prediction")

col1, col2, col3 = st.columns(3)
with col1:
    age_input = st.number_input("Age (years)", min_value=18, max_value=110, value=65, step=1)
    size_input = st.number_input("Tumor size (mm)", min_value=1, max_value=100, value=30, step=1)

with col2:
    loc_levels = ["(baseline)"] + artifact.get("location_levels", [])
    sel_loc = st.selectbox("Location", options=loc_levels, index=0)

with col3:
    epilepsi_in  = st.selectbox("Epilepsi", [0, 1], index=0)
    tryk_in       = st.selectbox("Tryksympt", [0, 1], index=0)
    focal_in      = st.selectbox("Focalsympt", [0, 1], index=0)
    calcified_in  = st.selectbox("Calcified", [0, 1], index=1)
    edema_in      = st.selectbox("Edema", [0, 1], index=0)

# -----------------------------
# OOD guardrails
# -----------------------------
ood_msgs = []
if age_input < train_ranges["age_min"] or age_input > train_ranges["age_max"]:
    ood_msgs.append(f"Age outside training range [{train_ranges['age_min']:.0f}, {train_ranges['age_max']:.0f}]")
if size_input < train_ranges["size_min"] or size_input > train_ranges["size_max"]:
    ood_msgs.append(f"Tumor size outside training range [{train_ranges['size_min']:.0f}, {train_ranges['size_max']:.0f}]")
if sel_loc != "(baseline)":
    chosen_dummy = f"location_{sel_loc}"
    if chosen_dummy not in feature_names:
        ood_msgs.append(f"Unseen location level: {sel_loc}")

# -----------------------------
# Predict
# -----------------------------
row_df = make_row(age_input, size_input, sel_loc if sel_loc != "(baseline)" else "BASELINE",
                  epilepsi_in, tryk_in, focal_in, calcified_in, edema_in, feature_names)

with st.spinner("Predicting..."):
    p = float(model.predict_proba(row_df)[:, 1][0])

risk_pct = 100.0 * p

# -----------------------------
# Reliability-based CI lookup
# -----------------------------
val_ci = None
val_obs = None
val_n = None
rel_bins = valB.get("reliability_bins", [])
if rel_bins:
    chosen = None
    for b in rel_bins:
        if (p >= b["p_min"]) and (p < b["p_max"]):
            chosen = b
            break
    if chosen is None:
        diffs = [abs(p - b["mean_pred"]) for b in rel_bins]
        chosen = rel_bins[int(np.argmin(diffs))]
    val_ci = (chosen["ci_low"], chosen["ci_high"])
    val_obs = chosen["obs_rate"]
    val_n   = chosen["n"]

# -----------------------------
# Render result
# -----------------------------
left, right = st.columns([1.2, 1])
with left:
    st.subheader("Estimated probability of intervention within 15 years")
    if val_ci:
        st.metric(
            label="Risk (calibrated)",
            value=f"{risk_pct:.1f}%",
            delta=f"95% CI {val_ci[0]*100:.1f}–{val_ci[1]*100:.1f}%"
        )
        if val_obs is not None:
            st.caption(f"External reliability: observed {val_obs*100:.1f}% (n={val_n}) in this risk band (Center B).")
    else:
        st.metric(label="Risk (calibrated)", value=f"{risk_pct:.1f}%")

    if p <= 0.05 and not ood_msgs:
        st.success("Eligible for very-low risk rule-out (≤5% and within training distribution).")
    elif p <= 0.05 and ood_msgs:
        st.warning("Probability is ≤5%, but inputs are outside training range. Use caution.")

with right:
    st.subheader("Input sanity / guardrails")
    if ood_msgs:
        for m in ood_msgs:
            st.error("OOD: " + m)
    else:
        st.info("All inputs within training distribution & known categories.")

# -----------------------------
# Model card
# -----------------------------
with st.expander("Model card / notes"):
    st.markdown(f"""
- **Model:** {artifact['model_type']}  
- **Calibration:** Isotonic (5-fold CV)  
- **Training:** Center A  
- **External validation (Center B):** AUC ≈ {valB.get('auc', None):.3f}, Brier ≈ {valB.get('brier', None):.3f}  
- **Use case:** Rule-out follow-up MRI for truly incidental meningioma when risk is very low.  
- **Caveats:** Out-of-distribution inputs reduce reliability; practice drift requires re-calibration; use alongside clinical judgment.
""")
