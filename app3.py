import joblib
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# Load artifact
# -----------------------------
ARTIFACT_PATH = "meningioma_rf_models.joblib"
artifact = joblib.load(ARTIFACT_PATH)

model_A = artifact["calibrated_model_A"]
model_AB = artifact["calibrated_model_AB"]
feature_names = artifact["feature_names"]

# -----------------------------
# Location mapping
# -----------------------------
location_map = {
    0: "infratentorial",
    1: "supratentorial",
    2: "skullbase",
    3: "convexity"
}

# -----------------------------
# Helper to build feature row
# -----------------------------
def make_row(age, size_mm, location_code, epilepsy, ich_sympt, focal, calcified, edema, feature_names):
    row = {
        "age": age,
        "tumorsize": size_mm,   # stored in mm
        "epilepsi": int(epilepsy),
        "tryksympt": int(ich_sympt),
        "focalsympt": int(focal),
        "calcified": int(calcified),
        "edema": int(edema),
    }
    row_df = pd.DataFrame([row])
    # add location dummies
    for c in feature_names:
        if c.startswith("location_"):
            row_df[c] = 0
    chosen_col = f"location_{location_map[location_code]}"
    if chosen_col in feature_names:
        row_df[chosen_col] = 1
    for c in feature_names:
        if c not in row_df.columns:
            row_df[c] = 0
    return row_df[feature_names]

# -----------------------------
# Helper: lookup calibration bin
# -----------------------------
def lookup_bins(p, bins):
    for b in bins:
        if b["p_min"] <= p <= b["p_max"]:
            return b
    return None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Meningioma Risk (15y intervention)", layout="wide")
st.title("Meningioma 15-Year Intervention Risk")

st.sidebar.header("Patient inputs")
age_input = st.sidebar.number_input("Age (years)", 18, 110, 65)
size_input = st.sidebar.number_input("Tumor size (mm)", 1, 100, 30)

# Location dropdown
sel_loc_code = st.sidebar.selectbox(
    "Location",
    options=list(location_map.keys()),
    format_func=lambda x: {0:"Infratentorial",1:"Supratentorial",2:"Skull base",3:"Convexity"}[x]
)

# Dichotomous variables (0=No, 1=Yes)
yes_no = {0: "No", 1: "Yes"}

epilepsy_in  = st.sidebar.selectbox("Epilepsy", [0,1], format_func=lambda x: yes_no[x])
ich_in       = st.sidebar.selectbox("Intracranial Hypertension Symptoms", [0,1], format_func=lambda x: yes_no[x])
focal_in     = st.sidebar.selectbox("Focal Neurologic Symptoms", [0,1], format_func=lambda x: yes_no[x])
calcified_in = st.sidebar.selectbox(">50% of tumor calcified", [0,1], format_func=lambda x: yes_no[x], index=1)
edema_in     = st.sidebar.selectbox("Edema", [0,1], format_func=lambda x: yes_no[x])

# -----------------------------
# Prediction
# -----------------------------
row_df = make_row(age_input, size_input, sel_loc_code, epilepsy_in, ich_in, focal_in, calcified_in, edema_in, feature_names)

p_A = float(model_A.predict_proba(row_df)[0, 1])
p_AB = float(model_AB.predict_proba(row_df)[0, 1])

bin_A = lookup_bins(p_A, artifact["validation_A"]["reliability_bins"])
bin_AB = lookup_bins(p_AB, artifact["validation_AB"]["reliability_bins"])
bin_B = lookup_bins(p_A, artifact["validation_B"]["reliability_bins"])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Center A model (trained only on A)")
    if bin_A:
        st.write(f"**Risk:** {p_A*100:.1f}% (95% CI {bin_A['ci_low']*100:.1f}–{bin_A['ci_high']*100:.1f}%)")
        st.caption(f"Observed in Center A: {bin_A['obs_rate']*100:.1f}% (n={bin_A['n']})")
    if bin_B:
        st.caption(f"Observed in Center B (external): {bin_B['obs_rate']*100:.1f}% (n={bin_B['n']})")

with col2:
    st.subheader("Pooled model (trained on A+B)")
    if bin_AB:
        st.write(f"**Risk:** {p_AB*100:.1f}% (95% CI {bin_AB['ci_low']*100:.1f}–{bin_AB['ci_high']*100:.1f}%)")
        st.caption(f"Observed in pooled A+B: {bin_AB['obs_rate']*100:.1f}% (n={bin_AB['n']})")

st.markdown("---")
st.subheader("Model description and interpretation")

valB = artifact["validation_B"]

st.markdown(f"""
The prediction tool is based on a Random Forest classifier with isotonic calibration. 
Two versions of the model are provided: one trained exclusively on patients from Center A, and one trained on the pooled cohort from Centers A and B. 
The Center A model is the primary reference, as it has been externally validated on an independent dataset from Center B. 
In this validation, the model achieved an AUC of {valB['auc']:.3f} and a Brier score of {valB['brier']:.3f}, indicating good discriminative performance and overall calibration.

Model calibration was assessed by dividing predictions into ten probability bins of equal size. 
Within each bin, the mean predicted probability of intervention was compared with the observed proportion of patients who underwent surgery. 
Ninety-five percent confidence intervals for these observed proportions were calculated using Wilson’s binomial method. 
The width of these intervals reflects the number of patients available in each bin; wider intervals occur where the number of patients is limited.

For an individual patient, the model outputs a predicted probability of intervention within 15 years. 
This probability should be interpreted alongside the 95% confidence interval, which reflects uncertainty in the calibration of the model within the relevant probability range. 
The application also reports the observed event rate among patients in the validation cohort who fell within the same predicted risk range, thereby grounding the estimate in empirical outcomes. 

The pooled model trained on Centers A and B is presented as a supplementary reference. 
While it incorporates a larger sample size and may generalize better to mixed populations, it does not have an independent external validation and should therefore be interpreted with more caution. 
In all cases, the estimates provided by this tool are intended to support—but not replace—clinical judgment in decision-making regarding follow-up of incidental meningioma.
""")


