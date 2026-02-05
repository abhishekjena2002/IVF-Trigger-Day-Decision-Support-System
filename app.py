# ============================================
# app.py
# IVF Trigger Day – Batch + Patient View App
# ============================================

import streamlit as st
import pandas as pd
import joblib

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="IVF Trigger Day Prediction",
    layout="centered"
)

st.title("IVF Trigger Day Prediction System")

# ============================================
# Load ML Artifacts
# ============================================
@st.cache_resource
def load_ml_artifacts():
    preprocessor = joblib.load("preprocessor_tree.pkl")
    model = joblib.load("ivf_xgboost_model.pkl")
    return preprocessor, model

preprocessor, model = load_ml_artifacts()

# ============================================
# Menu Tabs
# ============================================
tab1, tab2 = st.tabs(["📤 Batch Prediction", "👤 Patient View"])

# ============================================
# TAB 1: Batch Prediction
# ============================================
with tab1:

    st.subheader("📤 Upload New Patient Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        # Load file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully")

        # Normalize column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace("/", "_", regex=False)
        )

        # Store raw dataset
        st.session_state["raw_df"] = df.copy()

        # -----------------------------
        # Feature Engineering
        # -----------------------------
        df_fe = df.copy()

        df_fe["e2_per_follicle"] = (
            df_fe["estradiol_pg_ml"] /
            df_fe["follicle_count"].replace(0, 1)
        )

        df_fe["follicle_maturity_index"] = (
            df_fe["avg_follicle_size_mm"] *
            df_fe["follicle_count"]
        )

        df_fe["amh_group_normal_vs_high"] = (
            df_fe["amh_ng_ml"] >= 4.0
        ).astype(int)

        df_fe["follicle_size_17_20mm_flag"] = (
            (df_fe["avg_follicle_size_mm"] >= 17) &
            (df_fe["avg_follicle_size_mm"] <= 20)
        ).astype(int)

        MODEL_FEATURES = [
            "age",
            "day",
            "amh_ng_ml",
            "estradiol_pg_ml",
            "progesterone_ng_ml",
            "avg_follicle_size_mm",
            "follicle_count",
            "e2_per_follicle",
            "follicle_maturity_index",
            "amh_group_normal_vs_high",
            "follicle_size_17_20mm_flag"
        ]

        X = df_fe[MODEL_FEATURES]

        # Predict
        X_proc = preprocessor.transform(X)
        predictions = model.predict(X_proc)

        df["trigger_day_prediction"] = predictions.astype(int)

        # Save predicted data
        st.session_state["predicted_df"] = df.copy()

        st.subheader("📊 Batch Prediction Output")
        st.dataframe(df)

        st.download_button(
            label="⬇️ Download Prediction Results",
            data=df.to_csv(index=False),
            file_name="ivf_trigger_day_predictions.csv",
            mime="text/csv"
        )

# ============================================
# TAB 2: Patient View
# ============================================
with tab2:

    st.subheader("👤 Patient-Level Decision View")

    if "predicted_df" not in st.session_state:
        st.warning("Please upload and run prediction in Batch Prediction tab first.")
        st.stop()

    df = st.session_state["predicted_df"]

    if "patient_id" not in df.columns:
        st.error("Column `patient_id` is required in the dataset.")
        st.stop()

    patient_id = st.selectbox(
        "Select Patient ID",
        sorted(df["patient_id"].unique())
    )

    patient_data = df[df["patient_id"] == patient_id]

    st.markdown("### 📋 Patient Data (All Days)")
    st.dataframe(patient_data)

    # ----------------------------------------
    # CORRECT IVF DECISION LOGIC
    # ----------------------------------------
    patient_data["trigger_day_prediction"] = (
        patient_data["trigger_day_prediction"].astype(int)
    )

    # If ANY day is predicted as 1 → GOOD DAY EXISTS
    if patient_data["trigger_day_prediction"].max() == 1:
        final_decision = 1
    else:
        final_decision = 0

    # Debug (optional but useful)
    st.markdown("#### 🔎 Prediction Summary")
    st.write(patient_data["trigger_day_prediction"].value_counts())

    # ----------------------------------------
    # Final Decision Display
    # ----------------------------------------
    if final_decision == 1:
        st.markdown(
            """
            <div style="padding:20px;border-radius:10px;
                        background-color:#e6fffa;
                        border:2px solid green">
                <h2 style="color:green;text-align:center;">
                    ✅ GOOD DAY FOR TRIGGER
                </h2>
                <p style="text-align:center;">
                    At least one stimulation day is clinically suitable.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="padding:20px;border-radius:10px;
                        background-color:#ffe6e6;
                        border:2px solid red">
                <h2 style="color:red;text-align:center;">
                    ❌ NOT A GOOD DAY FOR TRIGGER
                </h2>
                <p style="text-align:center;">
                    No suitable stimulation day identified yet.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================
# Footer
# ============================================
st.markdown("---")
st.caption("© IVF Trigger Day Decision Support | Batch + Patient-Level Inference")
