import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Meningioma Risk (15y intervention)", layout="wide")
st.title("Meningioma 15-Year Intervention Risk")
st.caption("Random Forest (isotonic calibrated). Trained on Center A, validated on Center B. "
           "CIs from pooled A+B; observed rates from Center B and A+B.")

# -----------------------------
# Load artifact
# -----------------------------
@st.cache_resource
def load_artifact(path="meningioma_rf_model.joblib"):
    return joblib.load(path)

artifact = load_artifact()
model = artifact["calibrated_model"]
feature_names = artifact["feature_names"]
train_ranges = artifact["train_ranges"]
valB = artifact.get("validation_B", {})
valAB = artifact.get("validation_AB", {})

# -----------------------------
# Predict helper
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
    return row_df[feature_names]

# -----------------------------
# Bin lookup (safe)
# -----------------------------
def lookup_bins(prob, bins):
    if not bins: 
        return None
    chosen = None
    for b in bins:
        if prob >= b["p_min"] and prob <= b["p_max"]:
            chosen = b
            break
    if chosen is None:
        diffs = [abs(prob - b["mean_pred"]) for b in bins]
        chosen = bins[int(np.argmin(diffs))]
    return chosen

# -----------------------------
# UI inputs
# -----------------------------
st.header("Interactive prediction")

col1, col2, col3 = st.columns(3)
with col1:
    age_input = st.number_input("Age (years)", 18, 110, 65, 1)
    size_input = st.number_input("Tumor size (mm)", 1, 100, 30, 1)

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
# Predict
# -----------------------------
row_df = make_row(age_input, size_input,
                  sel_loc if sel_loc != "(baseline)" else "BASELINE",
                  epilepsi_in, tryk_in, focal_in, calcified_in, edema_in,
                  feature_names)

p = float(model.predict_proba(row_df)[:, 1][0])
risk_pct = 100 * p

# -----------------------------
# Lookup bins
# -----------------------------
bin_AB = lookup_bins(p, valAB.get("reliability_bins", []))  # pooled A+B for CI + obs
bin_B  = lookup_bins(p, valB.get("reliability_bins", []))   # external B for obs only

# -----------------------------
# Render results
# -----------------------------
st.subheader("Estimated probability of intervention within 15 years")

if bin_AB:
    ci_low = bin_AB['ci_low']*100
    ci_high = bin_AB['ci_high']*100
    st.write(f"**Risk: {risk_pct:.1f}% (95% CI {ci_low:.1f}–{ci_high:.1f}%)**")
else:
    st.write(f"**Risk: {risk_pct:.1f}%**")

# Always show both observed rates if available
if bin_B:
    st.caption(
        f"External reliability (Center B): observed {bin_B['obs_rate']*100:.1f}% "
        f"(n={bin_B['n']}) in this risk band."
    )

if bin_AB:
    st.caption(
        f"Pooled reliability (Centers A+B): observed {bin_AB['obs_rate']*100:.1f}% "
        f"(n={bin_AB['n']}) in this risk band."
    )

# -----------------------------
# Model card
# -----------------------------
with st.expander("Model card / notes"):
    st.markdown(f"""
- **Model:** {artifact['model_type']}  
- **Calibration:** Isotonic (5-fold CV)  
- **Training:** Center A  
- **External validation (Center B):** AUC ≈ {valB.get('auc', None):.3f}, Brier ≈ {valB.get('brier', None):.3f}  
- **Reliability:** CI + pooled observed rate from A+B; external observed rate from B only.  
- **Use case:** Rule-out follow-up MRI for incidental meningioma at very low risk.  
- **Caveats:** Out-of-distribution inputs reduce reliability; use alongside clinical judgment.
""")
