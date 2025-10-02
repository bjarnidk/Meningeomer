import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -------------------------
# Load artifact
# -------------------------
ARTIFACT_PATH = "meningioma_rf_model.joblib"
artifact = joblib.load(ARTIFACT_PATH)

model = artifact["calibrated_model"]
feature_names = artifact["feature_names"]
train_ranges = artifact["train_ranges"]

# -------------------------
# Helpers
# -------------------------
def make_row(age, size, location_value, epilepsi, tryk, focal, calcified, edema, feature_names):
    row = {
        "age": age,
        "tumorsize": size,
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
    return row_df[feature_names]

def lookup_bins(p, bins):
    for b in bins:
        if b["p_min"] <= p <= b["p_max"]:
            return b
    return None

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Meningioma Intervention Risk", layout="wide")
st.title("Meningioma 15-Year Intervention Risk")
st.caption("Random Forest with isotonic calibration. Trained on Center A, validated on Center B.")

# Input panel
col1, col2, col3 = st.columns(3)
with col1:
    age_input = st.number_input("Age (years)", min_value=18, max_value=110, value=65)
    size_input = st.number_input("Tumor size (mm)", min_value=1, max_value=100, value=30, step=1)

with col2:
    sel_loc = st.selectbox("Location", options=["(baseline)"] + artifact["location_levels"])
    epilepsi_in  = st.selectbox("Epilepsi", [0, 1], index=0)
    tryk_in       = st.selectbox("Tryksympt", [0, 1], index=0)

with col3:
    focal_in     = st.selectbox("Focalsympt", [0, 1], index=0)
    calcified_in = st.selectbox("Calcified", [0, 1], index=1)
    edema_in     = st.selectbox("Edema", [0, 1], index=0)

# Make row
row_df = make_row(age_input, size_input,
                  sel_loc if sel_loc != "(baseline)" else "BASELINE",
                  epilepsi_in, tryk_in, focal_in, calcified_in, edema_in, feature_names)

# Prediction
p = float(model.predict_proba(row_df)[:, 1][0])
risk_pct = p * 100

st.subheader("Predicted probability of intervention within 15 years")
st.write(f"**Risk:** {risk_pct:.1f}%")

# --- Show both Center A and Pooled A+B results ---
colA, colAB = st.columns(2)

# Center A
with colA:
    bin_A = lookup_bins(p, artifact["validation_A"]["reliability_bins"])
    st.markdown("### Center A (training cohort)")
    if bin_A:
        ci_low = bin_A["ci_low"] * 100
        ci_high = bin_A["ci_high"] * 100
        obs = bin_A["obs_rate"] * 100
        n = bin_A["n"]
        st.metric("Risk (with 95% CI)", f"{risk_pct:.1f}% ({ci_low:.1f}–{ci_high:.1f}%)")
        st.caption(f"Observed {obs:.1f}% in this risk band (n={n})")
    else:
        st.warning("No calibration bin found for Center A.")

# Pooled A+B
with colAB:
    bin_AB = lookup_bins(p, artifact["validation_AB"]["reliability_bins"])
    st.markdown("### Pooled Centers A+B")
    if bin_AB:
        ci_low = bin_AB["ci_low"] * 100
        ci_high = bin_AB["ci_high"] * 100
        obs = bin_AB["obs_rate"] * 100
        n = bin_AB["n"]
        st.metric("Risk (with 95% CI)", f"{risk_pct:.1f}% ({ci_low:.1f}–{ci_high:.1f}%)")
        st.caption(f"Observed {obs:.1f}% in this risk band (n={n})")
    else:
        st.warning("No calibration bin found for pooled A+B.")

# Model card
with st.expander("Model card"):
    st.markdown(f"""
    - **Model:** {artifact['model_type']}
    - **Training cohort:** Center A  
    - **External validation:** Center B  
    - **Reliability shown:** Both Center A (internal calibration) and pooled A+B  
    - **Inputs:** Age, tumor size (mm), location, epilepsy, pressure symptoms, focal symptoms, calcification, edema  
    - **Output:** Estimated 15-year probability of surgical intervention  
    - **Interpretation:** Use the risk estimate **plus the CI** and the observed rates for context. 
      If the CI is wide, clinical judgment is especially important.
    """)
