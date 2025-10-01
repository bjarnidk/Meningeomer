import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Meningioma 15y Intervention Risk", layout="wide")
st.title("Meningioma 15-Year Intervention Risk (Frozen Model)")
st.caption("Random Forest with isotonic calibration. Trained on Center A (tumor size in mm); validated on Center B.")

# -----------------------------
# Load artifact
# -----------------------------
ARTIFACT_PATH = Path("meningioma_rf_model.joblib")  # ensure this file is in the same folder
if not ARTIFACT_PATH.exists():
    st.error(f"Model artifact not found at {ARTIFACT_PATH.resolve()}. Upload the .joblib file to this folder.")
    st.stop()

artifact = joblib.load(ARTIFACT_PATH)
model = artifact["calibrated_model"]
feature_names = artifact["feature_names"]
train_ranges = artifact["train_ranges"]
location_levels = artifact["location_levels"]
valB = artifact["validation_B"]
feature_importances = artifact.get("feature_importances", None)

# Model expects mm
st.info("📏 Tumor size is expected in **millimeters (mm)** — this matches the training data.")

# -----------------------------
# Sidebar: policy settings
# -----------------------------
st.sidebar.header("Policy & Display")
policy_t = st.sidebar.slider("Decision threshold t (predict 'intervention' if p ≥ t)", 0.0, 1.0, 0.30, 0.01)
low_risk_cut = st.sidebar.slider("Very-low risk band (rule-out candidate)", 0.00, 0.20, 0.05, 0.01)

# -----------------------------
# Helpers
# -----------------------------
def make_row(age, size_mm, location_value, epilepsi, tryk, focal, calcified, edema, feature_names):
    """Build one-hot encoded row for prediction. size must be in mm (as per training)."""
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

    # create all possible location dummies seen in training; default 0
    for c in feature_names:
        if c.startswith("location_"):
            row_df[c] = 0

    # set chosen location dummy (if not baseline and exists)
    if location_value is not None and location_value != "(baseline)":
        chosen_col = f"location_{location_value}"
        if chosen_col in feature_names:
            row_df[chosen_col] = 1

    # ensure all features exist & correct order
    for c in feature_names:
        if c not in row_df.columns:
            row_df[c] = 0
    row_df = row_df[feature_names]
    return row_df

def tree_variance(model, row_df):
    """Compute variance across RF trees for one prediction (proxy for internal uncertainty)."""
    try:
        rf = model.base_estimator  # RF inside calibrator
        tree_preds = np.array([tree.predict_proba(row_df)[:, 1][0] for tree in rf.estimators_])
        return float(tree_preds.var()), float(tree_preds.std())
    except Exception:
        return None, None

# -----------------------------
# Performance summary (from artifact)
# -----------------------------
with st.expander("Validation summary (Center B)"):
    st.write(f"**AUC (B)**: {valB['auc']:.3f}  |  **Brier (B)**: {valB['brier']:.3f}")
    st.text("Confusion matrix @ t=0.5 (B):\n" + str(np.array(valB["confusion_matrix"])))
    st.text(valB["classification_report"])
    st.write("Threshold trade-offs (Center B):")
    st.dataframe(pd.DataFrame(valB["threshold_table"]))

# Optional: top features
if feature_importances:
    st.subheader("Top features (training model importance)")
    fi = pd.Series(feature_importances).sort_values(ascending=False)
    st.write(fi.head(12))

# -----------------------------
# Patient input UI
# -----------------------------
st.header("Patient risk calculator")

col1, col2, col3 = st.columns(3)
with col1:
    age_input = st.number_input("Age (years)", min_value=18, max_value=110, value=65, step=1)
    size_mm = st.number_input("Tumor size (mm)", min_value=1, max_value=200, value=30, step=1)
with col2:
    loc_options = ["(baseline)"] + location_levels
    sel_loc = st.selectbox("Location", options=loc_options, index=0)
with col3:
    epilepsi_in  = st.selectbox("Epilepsi",   [0, 1], index=0)
    tryk_in      = st.selectbox("Tryksympt",  [0, 1], index=0)
    focal_in     = st.selectbox("Focalsympt", [0, 1], index=0)
    calcified_in = st.selectbox("Calcified",  [0, 1], index=1)
    edema_in     = st.selectbox("Edema",      [0, 1], index=0)

# -----------------------------
# OOD guardrails
# -----------------------------
ood_msgs = []
if age_input < train_ranges["age_min"] or age_input > train_ranges["age_max"]:
    ood_msgs.append(f"Age outside training range [{train_ranges['age_min']:.0f}, {train_ranges['age_max']:.0f}] years")
if size_mm < train_ranges["size_min"] or size_mm > train_ranges["size_max"]:
    ood_msgs.append(f"Tumor size outside training range [{train_ranges['size_min']:.0f}, {train_ranges['size_max']:.0f}] mm")
if sel_loc != "(baseline)":
    chosen_dummy = f"location_{sel_loc}"
    if chosen_dummy not in feature_names:
        ood_msgs.append(f"Unseen location level: {sel_loc}")

# -----------------------------
# Build row & predict
# -----------------------------
row_df = make_row(age_input, size_mm, sel_loc, epilepsi_in, tryk_in, focal_in, calcified_in, edema_in, feature_names)
p = float(model.predict_proba(row_df)[:, 1][0])  # calibrated probability
risk_pct = p * 100
var, std = tree_variance(model, row_df)

# -----------------------------
# Risk display
# -----------------------------
if p <= low_risk_cut:
    band = "Very low"
elif p <= 0.15:
    band = "Low"
elif p <= 0.30:
    band = "Intermediate"
else:
    band = "High"

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Estimated probability of intervention within 15 years")
    st.metric(label="Risk (calibrated)", value=f"{risk_pct:.1f}%")
    if std is not None:
        st.caption(f"Model internal uncertainty (tree std): ±{std*100:.1f}%")
    st.write(f"**Risk band:** {band}  |  **Policy t:** {policy_t:.2f}  |  **Very-low cut:** ≤ {low_risk_cut:.2f}")

    pred_at_t = int(p >= policy_t)
    if pred_at_t == 1:
        st.warning("Model classification at chosen threshold: **Intervention likely** (≥ t).")
    else:
        st.success("Model classification at chosen threshold: **No intervention likely** (< t).")

    if p <= low_risk_cut and not ood_msgs:
        st.success("Eligible for **very-low risk rule-out** (≤ cut and within training distribution). Use clinical judgment.")
    elif p <= low_risk_cut and ood_msgs:
        st.warning("Probability is ≤ very-low cut, **but inputs are outside training range**. Use caution.")

with right:
    st.subheader("Input sanity / guardrails")
    if ood_msgs:
        for m in ood_msgs:
            st.error("OOD: " + m)
    else:
        st.info("All inputs within training distribution & known categories.")

# -----------------------------
# Calibration curve display
# -----------------------------
with st.expander("Calibration curve (Center B)"):
    bins = valB.get("calibration_bins", [])
    if bins:
        mp = [b["mean_pred"] for b in bins]
        fp = [b["frac_pos"] for b in bins]
        fig, ax = plt.subplots()
        ax.plot(mp, fp, "o-", label="Observed")
        ax.plot([0,1],[0,1],"--", color="gray", label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction of positives")
        ax.set_title("Calibration (Center B)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("No calibration bins stored in artifact.")

# -----------------------------
# Model card / notes
# -----------------------------
with st.expander("Model card / notes"):
    st.markdown(f"""
- **Model:** {artifact['model_type']}
- **Training:** Center A only (frozen). Tumor size in **mm**.
- **External validation (Center B):** AUC ≈ {valB['auc']:.3f}, Brier ≈ {valB['brier']:.3f}
- **Variables:** age, tumorsize (mm, continuous); location (categorical); epilepsi, tryksympt, focalsympt, calcified, edema (binary).
- **Calibration:** Isotonic (5-fold CV).
- **Uncertainty:** 
    - Internal model disagreement (tree std) shown per patient.
    - Historical calibration curve from Center B included for reliability.
- **Intended use:** Rule-out follow-up MRI when risk is very low (e.g., ≤ {low_risk_cut:.2f}). Use alongside clinical judgment.
- **Guardrails:** OOD warnings for age/size out of range or unseen location.
""")
