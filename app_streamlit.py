# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBClassifier
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, RocCurveDisplay
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------- SAFE DIRECTORY SETUP ----------
try:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if app_dir:
        os.chdir(app_dir)
except Exception as e:
    print(f"[WARN] Working directory not changed: {e}")

# ---------- STREAMLIT CONFIG ----------
st.set_page_config(layout="wide", page_title="GeneWise - SNP Risk Detector")
st.title("🧬 GeneWise — SNP-based Genetic Risk Detector")
st.caption("A compact Explainable AI system for SNP-based genetic disorder prediction.")

# ---------- PATHS ----------
CLEANED_CSV = "cleaned_snp_dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

LR_PATH = os.path.join(MODEL_DIR, "lr_model.joblib")
XGB_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.joblib")

# ---------- HELPERS ----------
def load_data(path):
    return pd.read_csv(path)

def build_encoders_and_scale(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            cats = list(X[col].astype(str).unique())
            mapping = {cat: i for i, cat in enumerate(cats)}
            encoders[col] = mapping
            X[col] = X[col].astype(str).map(mapping).fillna(0).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.values, encoders, scaler, X.columns.tolist()

def apply_encoders_and_scale(df, encoders, scaler, feature_cols):
    X = df.copy()
    missing = [c for c in feature_cols if c not in X.columns]
    for c in missing:
        X[c] = 0
    X = X[feature_cols]
    for col, mapping in encoders.items():
        if col in X.columns:
            X[col] = X[col].astype(str).map(mapping).fillna(0).astype(float)
    X_scaled = scaler.transform(X)
    return X_scaled

# ---------- TRAINING ----------
def train_and_cache_models(df, target_col="Target"):
    X_scaled, y, encoders, scaler, feature_cols = build_encoders_and_scale(df, target_col)

    unique, counts = np.unique(y, return_counts=True)
    label_dist = dict(zip(unique, counts))
    st.write(f"🧩 Label distribution: {label_dist}")

    stratify_param = y if len(unique) > 1 and min(counts) >= 2 else None
    if stratify_param is None:
        st.warning("⚠️ Using random split (class imbalance detected).")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    joblib.dump(lr, LR_PATH)
    joblib.dump(xgb, XGB_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)

    return lr, xgb, scaler, encoders, feature_cols, X_test, y_test, X_train, y_train

# ---------- EVALUATION ----------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None and len(np.unique(y_test)) > 1 else None
    return {"accuracy": acc, "auc": auc, "y_pred": y_pred, "y_proba": y_proba}

# ---------- DATASET ----------
st.header("📂 Dataset Setup")
if os.path.exists(CLEANED_CSV):
    df = load_data(CLEANED_CSV)
    st.success(f"✅ Dataset loaded ({df.shape[0]} rows, {df.shape[1]} cols)")
else:
    uploaded = st.file_uploader("Upload dataset (.csv)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df.to_csv(CLEANED_CSV, index=False)
        st.success("✅ Dataset uploaded and saved.")
    else:
        st.stop()

target_col = "Target" if "Target" in df.columns else df.columns[-1]

# ---------- TRAIN ----------
st.header("1️⃣ Train / Load Models")
if all(os.path.exists(p) for p in [LR_PATH, XGB_PATH, SCALER_PATH, ENCODERS_PATH]):
    st.success("📦 Models found in cache.")
    retrain = st.button("🔁 Retrain Models")
else:
    retrain = st.button("🚀 Train Models")

if retrain:
    with st.spinner("Training models..."):
        lr, xgb, scaler, encoders, feature_cols, X_test, y_test, X_train, y_train = train_and_cache_models(df, target_col)
    st.success("✅ Training complete!")

if all(os.path.exists(p) for p in [LR_PATH, XGB_PATH, SCALER_PATH, ENCODERS_PATH]):
    lr = joblib.load(LR_PATH)
    xgb = joblib.load(XGB_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_cols = df.drop(columns=[target_col]).columns.tolist()

# ---------- TWO-COLUMN EVALUATION ----------
st.header("2️⃣ Model Evaluation Dashboard")
with st.expander("📊 Show compact dashboard"):
    X_scaled, y, _, _, _ = build_encoders_and_scale(df, target_col)
    stratify_param = y if len(np.unique(y)) > 1 and min(np.bincount(y.astype(int))) >= 2 else None
    if stratify_param is None:
        st.warning("⚠️ Using random split for evaluation.")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=stratify_param)

    eval_lr = evaluate_model(lr, X_test, y_test)
    eval_xgb = evaluate_model(xgb, X_test, y_test)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("### 🧮 Model Performance")
        st.write(f"**Logistic Regression**  \nAccuracy: `{eval_lr['accuracy']:.4f}`  \nAUC: `{eval_lr['auc'] or 'N/A'}`")
        st.text(classification_report(y_test, eval_lr['y_pred']))

        st.write(f"**XGBoost**  \nAccuracy: `{eval_xgb['accuracy']:.4f}`  \nAUC: `{eval_xgb['auc'] or 'N/A'}`")
        st.text(classification_report(y_test, eval_xgb['y_pred']))

    with col2:
        st.markdown("### 📈 ROC Curves")
        try:
            fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
            RocCurveDisplay.from_estimator(lr, X_test, y_test, ax=ax, name="LR")
            RocCurveDisplay.from_estimator(xgb, X_test, y_test, ax=ax, name="XGB")
            ax.set_title("ROC Curves", fontsize=8)
            ax.tick_params(axis='both', labelsize=6)
            st.pyplot(fig, clear_figure=True, use_container_width=False)
        except Exception:
            st.warning("ROC curve skipped — single class detected.")

# ---------- PREDICTION ----------
st.header("3️⃣ Predict New SNP Samples")
uploaded = st.file_uploader("Upload SNP data (.csv) for prediction", type=["csv"])
if uploaded:
    new_df = pd.read_csv(uploaded)
    st.dataframe(new_df.head(), height=150)

    X_new = apply_encoders_and_scale(new_df, encoders, scaler, feature_cols)
    model_choice = st.selectbox("Choose Model", ["LogisticRegression", "XGBoost"])
    model = lr if model_choice == "LogisticRegression" else xgb

    probs = model.predict_proba(X_new)[:, 1] if hasattr(model, "predict_proba") else [0]*len(X_new)
    preds = model.predict(X_new)

    results = new_df.copy()
    results["Predicted_Label"] = preds
    results["Predicted_Probability"] = probs
    st.dataframe(results, height=180)

    # ---------- SIDE-BY-SIDE SHAP ----------
    st.subheader("4️⃣ Explainability (Compact View)")
    col1, col2 = st.columns([1, 1])
    try:
        import shap
        with col1:
            plt.figure(figsize=(3, 2), dpi=100)
            explainer = shap.Explainer(model, shap.sample(X_new, min(50, len(X_new))))
            shap_values = explainer(X_new)
            shap.summary_plot(shap_values, show=False, plot_size=(3,2))
            st.pyplot(bbox_inches="tight", use_container_width=False)
        with col2:
            st.info("Top SHAP contributors visualized on the left ➡️")
    except Exception as e:
        st.info(f"⚠️ SHAP skipped: {e}")

st.markdown("---")
st.caption("GeneWise Streamlit App — Compact two-column dashboard for SNP-based genetic risk prediction.")
