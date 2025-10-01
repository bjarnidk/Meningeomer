import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix, classification_report

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Meningioma Risk (15y intervention)", layout="wide")
st.title("Meningioma 15-Year Intervention Risk")
st.caption("Random Forest (isotonic calibrated). Train on A+B, validate on B. Built for rule-out decisions with guardrails.")

# -----------------------------
# Helper: robust binary mapping
# -----------------------------
def to_binary_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    mapping = {
        "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1, "ja": 1, "sand": 1,
        "no": 0, "n": 0, "false": 0, "f": 0, "0": 0, "nej": 0
    }
    return s.astype(str).str.strip().str.lower().map(mapping)

# -----------------------------
# Sidebar: data upload & options
# -----------------------------
st.sidebar.header("1) Data")
fileA = st.sidebar.file_uploader("Upload Center A CSV (CSV1.csv)", type=["csv"])
fileB = st.sidebar.file_uploader("Upload Center B CSV (CSV2.csv)", type=["csv"])

st.sidebar.header("2) Model options")
n_estimators = st.sidebar.slider("RandomForest trees", 200, 1000, 500, 50)
max_depth = st.sidebar.selectbox("max_depth", [5, 7, 9, None], index=0)
min_samples_leaf = st.sidebar.selectbox("min_samples_leaf", [1, 2, 5, 10], index=0)
calibration_folds = st.sidebar.slider("Calibration folds", 3, 10, 5)
compute_ci = st.sidebar.checkbox("Compute 95% CI via bootstrap (slower)", value=False)
n_boot = st.sidebar.slider("Bootstrap resamples (CI)", 50, 500, 200, 50, disabled=not compute_ci)

st.sidebar.header("3) Policy threshold")
policy_t = st.sidebar.slider("Decision threshold (predict 'intervention' if p ≥ t)", 0.0, 1.0, 0.30, 0.01)
low_risk_cut = st.sidebar.slider("Very-low risk band (rule-out candidate)", 0.00, 0.20, 0.05, 0.01)

# Variables
continuous = ["age", "tumorsize"]
categorical = ["location"]              # unordered
binary      = ["epilepsi", "tryksympt", "focalsympt", "calcified", "edema"]
target      = "intervention"
required_cols = continuous + categorical + binary + [target]

# -----------------------------
# Cache: load & preprocess
# -----------------------------
@st.cache_data(show_spinner=True)
def load_df(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower()
    return df

@st.cache_data(show_spinner=True)
def preprocess(dfA, dfB):
    dfA = dfA.copy(); dfA.columns = dfA.columns.str.lower()
    dfB = dfB.copy(); dfB.columns = dfB.columns.str.lower()

    missA = [c for c in required_cols if c not in dfA.columns]
    missB = [c for c in required_cols if c not in dfB.columns]
    if missA or missB:
        raise ValueError(f"Missing columns — A: {missA} | B: {missB}")

    for col in binary + [target]:
        dfA[col] = to_binary_series(dfA[col]).astype(float)
        dfB[col] = to_binary_series(dfB[col]).astype(float)

    for col in continuous:
        dfA[col] = pd.to_numeric(dfA[col], errors="coerce")
        dfB[col] = pd.to_numeric(dfB[col], errors="coerce")

    dfA = dfA.dropna(subset=required_cols).reset_index(drop=True)
    dfB = dfB.dropna(subset=required_cols).reset_index(drop=True)

    for col in binary + [target]:
        dfA[col] = dfA[col].astype(int)
        dfB[col] = dfB[col].astype(int)

    X_A_raw = dfA[continuous + categorical + binary].copy()
    y_A = dfA[target].values
    X_B_raw = dfB[continuous + categorical + binary].copy()
    y_B = dfB[target].values

    X_all = pd.concat([X_A_raw, X_B_raw], axis=0)
    X_all = pd.get_dummies(X_all, columns=categorical, drop_first=True)

    X_A = X_all.iloc[:len(dfA), :].reset_index(drop=True)
    X_B = X_all.iloc[len(dfA):, :].reset_index(drop=True)

    train_ranges = {
        "age_min": float(dfA["age"].min()),
        "age_max": float(dfA["age"].max()),
        "size_min": float(dfA["tumorsize"].min()),
        "size_max": float(dfA["tumorsize"].max()),
        "location_levels": sorted([c for c in X_A.columns if c.startswith("location_")])
    }
    feature_names = X_A.columns.tolist()
    return dfA, dfB, X_A, y_A, X_B, y_B, feature_names, train_ranges

# -----------------------------
# Fit calibrated RF
# -----------------------------
@st.cache_resource(show_spinner=True)
def fit_calibrated_rf(X_train, y_train, n_estimators, max_depth, min_leaf, calib_folds):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=calib_folds, shuffle=True, random_state=42)
    cal = CalibratedClassifierCV(rf, method="isotonic", cv=cv)
    cal.fit(X_train, y_train)
    return cal

@st.cache_resource(show_spinner=True)
def fit_on_A_then_eval_B(X_A, y_A, X_B, y_B, n_estimators, max_depth, min_leaf, calib_folds):
    rfA = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=calib_folds, shuffle=True, random_state=42)
    calA = CalibratedClassifierCV(rfA, method="isotonic", cv=cv)
    calA.fit(X_A, y_A)
    probB = calA.predict_proba(X_B)[:, 1]
    predB = (probB >= 0.5).astype(int)
    aucB = roc_auc_score(y_B, probB)
    brierB = brier_score_loss(y_B, probB)
    cm = confusion_matrix(y_B, predB)
    report = classification_report(y_B, predB, digits=2)
    return aucB, brierB, cm, report

# =========================================================
# PIPELINE: only proceed when both files uploaded
# =========================================================
if not (fileA and fileB):
    st.info("Upload Center A and Center B CSVs in the sidebar to begin.")
    st.stop()

try:
    dfA_raw = load_df(fileA)
    dfB_raw = load_df(fileB)
    dfA, dfB, X_A, y_A, X_B, y_B, feature_names, train_ranges = preprocess(dfA_raw, dfB_raw)

    # Store into session state (prevents NameError on reruns)
    st.session_state["feature_names"] = feature_names
    st.session_state["train_ranges"] = train_ranges
    st.session_state["X_A"] = X_A; st.session_state["y_A"] = y_A
    st.session_state["X_B"] = X_B; st.session_state["y_B"] = y_B

except Exception as e:
    st.error(f"Preprocess error: {e}")
    st.stop()

# Train final model on pooled data
X_pool = pd.concat([X_A, X_B], axis=0).reset_index(drop=True)
y_pool = np.concatenate([y_A, y_B], axis=0)
model = fit_calibrated_rf(X_pool, y_pool, n_estimators, max_depth, min_samples_leaf, calibration_folds)

# Show historical external performance (A→B)
aucB, brierB, cmB, repB = fit_on_A_then_eval_B(X_A, y_A, X_B, y_B, n_estimators, max_depth, min_samples_leaf, calibration_folds)
with st.expander("Performance summary (Center B external validation using A-only training)"):
    st.write(f"**AUC (B)**: {aucB:.3f}  |  **Brier (B)**: {brierB:.3f}")
    st.text("Confusion matrix @ t=0.5 (B):\n" + str(cmB))
    st.text(repB)

# -----------------------------
# Predict helpers
# -----------------------------
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

    # create all possible location dummies seen in training; default 0
    for c in feature_names:
        if c.startswith("location_"):
            row_df[c] = 0

    # set chosen location dummy (if not baseline and exists)
    if location_value is not None and location_value != "(baseline)":
        chosen_col = f"location_{location_value}"
        if chosen_col in feature_names:
            row_df[chosen_col] = 1

    # ensure all features exist & order columns
    for c in feature_names:
        if c not in row_df.columns:
            row_df[c] = 0
    row_df = row_df[feature_names]
    return row_df

def predict_with_ci(model, row_df, X_train, y_train, n_boot=200):
    p = float(model.predict_proba(row_df)[:, 1][0])
    if n_boot <= 0:
        return p, None
    ps = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, len(X_train), len(X_train))
        Xb = X_train.iloc[idx]
        yb = y_train[idx]
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            class_weight="balanced", random_state=int(rng.integers(0, 10_000)), n_jobs=-1
        )
        cv = StratifiedKFold(n_splits=max(3, min(5, len(np.unique(yb)))), shuffle=True, random_state=42)
        cal = CalibratedClassifierCV(rf, method="isotonic", cv=cv)
        cal.fit(Xb, yb)
        ps.append(float(cal.predict_proba(row_df)[:, 1][0]))
    low, high = np.percentile(ps, [2.5, 97.5])
    return p, (float(low), float(high))

# -----------------------------
# UI – Patient sliders / inputs
# -----------------------------
st.header("Interactive prediction")

feature_names = st.session_state["feature_names"]
train_ranges = st.session_state["train_ranges"]

col1, col2, col3 = st.columns(3)

with col1:
    age_input = st.number_input("Age (years)", min_value=18, max_value=110, value=65, step=1)
    size_input = st.number_input("Tumor size (cm)", min_value=0.1, max_value=10.0, value=3.0, step=0.1, format="%.1f")

with col2:
    loc_levels = ["(baseline)"] + [c.replace("location_", "") for c in train_ranges.get("location_levels", [])]
    sel_loc = st.selectbox("Location (category)", options=loc_levels, index=0)

with col3:
    epilepsi_in  = st.selectbox("Epilepsi", [0, 1], index=0)
    tryk_in       = st.selectbox("Tryksympt", [0, 1], index=0)
    focal_in      = st.selectbox("Focalsympt", [0, 1], index=0)
    calcified_in  = st.selectbox("Calcified", [0, 1], index=1)  # default calcified
    edema_in      = st.selectbox("Edema", [0, 1], index=0)

# OOD guardrails
ood_msgs = []
if age_input < train_ranges["age_min"] or age_input > train_ranges["age_max"]:
    ood_msgs.append(f"Age outside training range [{train_ranges['age_min']:.0f}, {train_ranges['age_max']:.0f}]")
if size_input < train_ranges["size_min"] or size_input > train_ranges["size_max"]:
    ood_msgs.append(f"Tumor size outside training range [{train_ranges['size_min']:.1f}, {train_ranges['size_max']:.1f}]")
if sel_loc != "(baseline)":
    chosen_dummy = f"location_{sel_loc}"
    if chosen_dummy not in feature_names:
        ood_msgs.append(f"Unseen location level: {sel_loc}")

# Build row & predict
row_df = make_row(
    age_input, size_input, sel_loc,
    epilepsi_in, tryk_in, focal_in, calcified_in, edema_in,
    feature_names
)

X_pool = pd.concat([st.session_state["X_A"], st.session_state["X_B"]], axis=0).reset_index(drop=True)
y_pool = np.concatenate([st.session_state["y_A"], st.session_state["y_B"]], axis=0)

with st.spinner("Predicting..."):
    if compute_ci:
        p, ci = predict_with_ci(model, row_df, X_pool, y_pool, n_boot=n_boot)
    else:
        p = float(model.predict_proba(row_df)[:, 1][0])
        ci = None

# Render result
risk_pct = 100.0 * p
band = ("Very low", "Low", "Intermediate", "High")
if p <= low_risk_cut:
    band_sel = band[0]
elif p <= 0.15:
    band_sel = band[1]
elif p <= 0.30:
    band_sel = band[2]
else:
    band_sel = band[3]

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Estimated probability of intervention within 15 years")
    if ci is not None:
        st.metric(label="Risk (calibrated)", value=f"{risk_pct:.1f}%", delta=f"95% CI {ci[0]*100:.1f}–{ci[1]*100:.1f}%")
    else:
        st.metric(label="Risk (calibrated)", value=f"{risk_pct:.1f}%")
    st.write(f"**Risk band:** {band_sel}  |  **Policy t:** {policy_t:.2f}  |  **Very-low cut:** ≤ {low_risk_cut:.2f}")

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

# Model card / notes
with st.expander("Model card / notes"):
    st.markdown(f"""
- **Model:** Random Forest (n_estimators={n_estimators}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf})  
- **Calibration:** Isotonic, {calibration_folds}-fold CV  
- **Training:** pooled Centers A+B (this session’s uploads)  
- **External validation (historical, A→B):** AUC ≈ {aucB:.3f}, Brier ≈ {brierB:.3f}  
- **Use case:** Rule-out follow-up MRI for truly incidental meningioma when risk is very low (e.g., ≤ {low_risk_cut:.2f}).  
- **Caveats:** Out-of-distribution inputs reduce reliability; practice drift requires re-calibration; use alongside clinical judgment.
""")
