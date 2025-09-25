import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

# -----------------------------
# 1) FIXED CORE MODEL (Center A)
#    coefficients from your last fit:
#    logit = b0 + b_age*Alder + b_edema*Peritumorodem + b_size*TumorSize_cm + b_calc*Calcifikation
# -----------------------------
COEF = {
    "const":         3.085418,
    "Alder":        -0.091704,
    "Peritumorodem": 1.435433,
    "TumorSize_cm":  0.084750,
    "Calcifikation": -1.045429,
}
PREDICTORS = ["Alder", "Peritumorodem", "TumorSize_cm", "Calcifikation"]
TARGET = "intervention"
DEFAULT_POLICY_CUTOFF = 0.05  # you can override via UI

st.set_page_config(page_title="Meningioma NO-MRI Policy Tool", page_icon="🧠", layout="wide")

st.title("🧠 Meningioma Surgery Risk & NO-MRI Policy (Core Model)")
st.caption("Core predictors: Age, Edema, Max diameter, Calcification. Trained on Center A; optional recalibration on Center A for policy threshold; optional validation on Center B.")

# -----------------------------
# Utility functions
# -----------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def predict_logistic(df: pd.DataFrame) -> np.ndarray:
    z = (COEF["const"]
         + COEF["Alder"]         * df["Alder"].astype(float)
         + COEF["Peritumorodem"] * df["Peritumorodem"].astype(float)
         + COEF["TumorSize_cm"]  * df["TumorSize_cm"].astype(float)
         + COEF["Calcifikation"] * df["Calcifikation"].astype(float))
    return sigmoid(z)

def choose_threshold_to_observed_risk(y_true: np.ndarray, p_pred: np.ndarray, target_obs: float) -> float:
    order = np.argsort(p_pred)
    y_sorted = y_true[order]
    p_sorted = p_pred[order]
    for t in np.unique(p_sorted):
        mask = p_sorted <= t
        if mask.sum() == 0: 
            continue
        obs_rate = y_sorted[mask].mean()
        if obs_rate <= target_obs:
            return float(t)
    # fallback: smallest prob (extremely conservative)
    return float(np.min(p_pred))

def policy_eval(y_true: np.ndarray, p_pred: np.ndarray, threshold: float):
    mask = p_pred <= threshold
    n = len(y_true)
    spared = int(mask.sum())
    obs_rate = float(np.nan) if spared == 0 else float(y_true[mask].mean())
    missed = 0 if spared == 0 else int(y_true[mask].sum())
    out = {
        "Total N": n,
        "Spared": f"{spared}/{n} ({(spared/n*100):.1f}%)",
        "Observed surgery rate in NO-FOLLOW": None if np.isnan(obs_rate) else obs_rate,
        "Missed surgeries (NO-FOLLOW)": missed
    }
    return out

def clean_single_meningioma(df: pd.DataFrame) -> pd.DataFrame:
    # If present, filter to single meningioma and drop the column
    if "Antalmeningeomer" in df.columns:
        df = df[df["Antalmeningeomer"] <= 1].copy()
        df.drop(columns=["Antalmeningeomer"], inplace=True)
    return df

def ensure_max_diameter(df: pd.DataFrame) -> pd.DataFrame:
    # If TumorSize_cm not present, but length/width exist, create max diameter
    if "TumorSize_cm" not in df.columns:
        needs = {"Tumorlengthcm", "Tumorwidthcm"}
        if needs.issubset(df.columns):
            df["TumorSize_cm"] = df[["Tumorlengthcm","Tumorwidthcm"]].max(axis=1)
        else:
            raise ValueError("Dataset must include TumorSize_cm or both Tumorlengthcm and Tumorwidthcm.")
    return df

# -----------------------------
# LEFT: Single-patient calculator
# -----------------------------
left, right = st.columns([1,1.3], gap="large")

with left:
    st.subheader("🧍 Single-patient risk calculator")

    age = st.number_input("Age (years)", min_value=18, max_value=110, value=80, step=1)
    size = st.number_input("Max tumor diameter (cm)", min_value=0.0, max_value=12.0, value=2.0, step=0.1, format="%.1f")
    edema = st.selectbox("Peritumoral edema", options=["No (0)","Yes (1)"], index=0)
    calc  = st.selectbox("Calcification",     options=["No (0)","Yes (1)"], index=1)

    df_one = pd.DataFrame([{
        "Alder": age,
        "Peritumorodem": 1 if "Yes" in edema else 0,
        "TumorSize_cm": size,
        "Calcifikation": 1 if "Yes" in calc else 0
    }])
    prob_raw = float(predict_logistic(df_one)[0])
    st.metric("Predicted surgery probability", f"{prob_raw*100:.1f}%")

    st.markdown("---")
    st.caption("Policy cutoff: choose a fixed probability threshold or derive it from Center A (upload below).")
    policy_mode = st.radio("Policy cutoff mode", ["Fixed cutoff (e.g., 5%)", "Derive from Center A (≤5% observed)"], horizontal=False)

    cutoff = DEFAULT_POLICY_CUTOFF
    iso = None
    derived_t = None

# -----------------------------
# RIGHT: Upload data for calibration/evaluation/batch
# -----------------------------
with right:
    st.subheader("📤 Optional: Upload Center A and/or B for calibration & evaluation")

    upA = st.file_uploader("Upload Center A CSV (to derive ≤5% cutoff & optional calibration)", type=["csv"], key="upA")
    upB = st.file_uploader("Upload Center B CSV (to evaluate external performance)", type=["csv"], key="upB")
    st.caption("Required columns: Alder, Peritumorodem, Calcifikation, TumorSize_cm (or Tumorlengthcm & Tumorwidthcm), intervention.")

    if policy_mode == "Derive from Center A (≤5% observed)" and upA is None:
        st.info("Upload Center A to derive the ≤5% cutoff. Otherwise switch to a fixed cutoff below.")

    if policy_mode == "Fixed cutoff (e.g., 5%)":
        cutoff = st.slider("Policy cutoff (probability)", 0.0, 0.20, value=DEFAULT_POLICY_CUTOFF, step=0.005, format="%.3f")

    # If Center A uploaded, derive isotonic calibration + threshold
    if upA is not None:
        dfA = pd.read_csv(upA)
        dfA = clean_single_meningioma(dfA)
        dfA = ensure_max_diameter(dfA)

        # keep only complete cases
        need_cols = PREDICTORS + [TARGET]
        dfA = dfA.dropna(subset=need_cols).copy()

        # split 50/50 (calibration split is the "evaluation half")
        rng = np.random.RandomState(42)
        idx = np.arange(len(dfA))
        rng.shuffle(idx)
        mid = len(dfA)//2
        idx_tr, idx_cal = idx[:mid], idx[mid:]

        X_tr = dfA.iloc[idx_tr][PREDICTORS].astype(float)
        y_tr = dfA.iloc[idx_tr][TARGET].astype(int).values
        X_cal = dfA.iloc[idx_cal][PREDICTORS].astype(float)
        y_cal = dfA.iloc[idx_cal][TARGET].astype(int).values

        # raw model (fixed coefficients) → raw probs
        p_cal_raw = predict_logistic(X_cal)

        # isotonic calibration using y_cal
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal_raw, y_cal)
        p_cal = iso.transform(p_cal_raw)

        auc_A = roc_auc_score(y_cal, p_cal)
        brier_A = brier_score_loss(y_cal, p_cal)
        st.success(f"Center A calibration half — AUC {auc_A:.3f}, Brier {brier_A:.3f}")

        # derive threshold for ≤5% observed risk
        derived_t = choose_threshold_to_observed_risk(y_cal, p_cal, 0.05)
        st.info(f"Derived Center-A policy threshold (≤5% observed): t = {derived_t:.4f}")

        if policy_mode == "Derive from Center A (≤5% observed)":
            cutoff = derived_t

        # Policy eval on A
        utilA = policy_eval(y_cal, p_cal, cutoff)
        st.write("**Policy on Center A (calibration half):**")
        st.json(utilA)

    # External validation on Center B (if uploaded)
    if upB is not None:
        dfB = pd.read_csv(upB)
        dfB = clean_single_meningioma(dfB)
        dfB = ensure_max_diameter(dfB)
        need_cols = PREDICTORS + [TARGET]
        dfB = dfB.dropna(subset=need_cols).copy()

        pB_raw = predict_logistic(dfB[PREDICTORS].astype(float))
        pB = iso.transform(pB_raw) if iso is not None else pB_raw  # use A-based calibration if available

        auc_B = roc_auc_score(dfB[TARGET].astype(int).values, pB)
        brier_B = brier_score_loss(dfB[TARGET].astype(int).values, pB)
        st.success(f"Center B — AUC {auc_B:.3f}, Brier {brier_B:.3f}")

        utilB = policy_eval(dfB[TARGET].astype(int).values, pB, cutoff)
        st.write("**Policy on Center B (using same cutoff):**")
        st.json(utilB)

        # Batch CSV with predictions + policy label
        out = dfB.copy()
        out["p_pred"] = pB
        out["policy"] = np.where(out["p_pred"] <= cutoff, "NO_FOLLOW", "FOLLOW")
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Center B predictions", data=csv, file_name="centerB_predictions_core_model.csv", mime="text/csv")

# -----------------------------
# Single-patient final decision block
# -----------------------------
with left:
    # If isotonic is available (A uploaded), use it for the single patient
    if iso is not None:
        prob_calibrated = float(iso.transform([prob_raw])[0])
        st.caption(f"Calibrated (Center-A isotonic) probability: **{prob_calibrated*100:.1f}%**")
        use_prob = prob_calibrated
    else:
        use_prob = prob_raw

    decision = "NO_FOLLOW (skip MRI)" if use_prob <= cutoff else "FOLLOW (MRI)"
    st.markdown(f"### Decision: **{decision}**")
    st.caption(f"Cutoff used: {cutoff:.4f}  •  Model: logit = {COEF['const']:+.3f} "
               f"+ {COEF['Alder']:+.3f}·Age + {COEF['Peritumorodem']:+.3f}·Edema "
               f"+ {COEF['TumorSize_cm']:+.3f}·MaxDiam + {COEF['Calcifikation']:+.3f}·Calcification")
