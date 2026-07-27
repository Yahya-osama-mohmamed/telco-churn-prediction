"""
Streamlit Web Dashboard — Interactive UI for the churn model.

Provides:
- Sidebar manual data entry form for single predictions
- Probability gauge chart and SHAP waterfall plot for interpretation
- Batch prediction via CSV upload
- Global feature importance view
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
from sklearn.pipeline import Pipeline

from src.config import FINAL_PIPELINE_PATH, MODEL_METADATA_PATH

# =============================================================================
# App Configuration
# =============================================================================
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurable via environment so docker-compose / Render can point the
# dashboard at the API container; falls back to local dev default.
API_URL = os.getenv("API_URL", "http://localhost:8000")


# =============================================================================
# Helper Functions
# =============================================================================

# Caches are keyed by artifact mtime: a model trained AFTER the dashboard
# started (or a retrain) is picked up automatically. Naively caching the
# loader itself would permanently pin the "no model yet" state.

@st.cache_resource
def _load_model_cached(mtime: float):
    return joblib.load(FINAL_PIPELINE_PATH)


def load_local_model():
    """Load model directly for SHAP explainability inside Streamlit."""
    try:
        if FINAL_PIPELINE_PATH.exists():
            return _load_model_cached(FINAL_PIPELINE_PATH.stat().st_mtime)
    except Exception as e:
        st.sidebar.error(f"Failed to load local model: {e}")
    return None


@st.cache_resource
def _load_threshold_cached(mtime: float) -> float:
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return float(metadata.get("decision_threshold", 0.5))


def load_decision_threshold() -> float:
    """Load the validation-tuned decision threshold (fallback 0.5)."""
    try:
        if MODEL_METADATA_PATH.exists():
            return _load_threshold_cached(MODEL_METADATA_PATH.stat().st_mtime)
    except Exception:
        pass
    return 0.5


def create_gauge_chart(probability: float, risk_level: str):
    """Create a Plotly gauge chart for churn probability."""
    if risk_level == "High":
        color = "#FF4B4B"  # Red
    elif risk_level == "Medium":
        color = "#FFA07A"  # Orange/Salmon
    else:
        color = "#00FA9A"  # Green
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Churn Probability (%)", 'font': {'size': 24}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 250, 154, 0.2)'},
                {'range': [30, 65], 'color': 'rgba(255, 160, 122, 0.2)'},
                {'range': [65, 100], 'color': 'rgba(255, 75, 75, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def predict_single_api(input_dict: dict):
    """Call the FastAPI endpoint for a single prediction."""
    try:
        response = requests.post(f"{API_URL}/predict", json=input_dict, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # Fallback to local prediction if API is down. The pipeline is
        # self-contained (feature engineering happens inside it).
        pipeline = load_local_model()
        if pipeline:
            df = pd.DataFrame([input_dict])
            prob = pipeline.predict_proba(df)[0][1]
            threshold = load_decision_threshold()
            risk = "High" if prob >= 0.65 else ("Medium" if prob >= 0.30 else "Low")
            return {
                "churn_prediction": int(prob >= threshold),
                "churn_probability": float(prob),
                "risk_level": risk,
                "_source": "local_fallback"
            }
        else:
            st.error(f"API Error and no local model available: {e}")
            return None


@st.cache_resource
def _load_shap_background_cached(_pipeline, train_mtime: float, model_mtime: float):
    from src.config import PROCESSED_TRAIN_FILE, TARGET
    df = pd.read_csv(PROCESSED_TRAIN_FILE).drop(columns=[TARGET], errors="ignore")
    transformer = Pipeline(_pipeline.steps[:-1])
    return np.asarray(transformer.transform(df))


def load_shap_background(pipeline):
    """Transformed training data used as background for LinearExplainer."""
    from src.config import PROCESSED_TRAIN_FILE
    try:
        if PROCESSED_TRAIN_FILE.exists() and FINAL_PIPELINE_PATH.exists():
            return _load_shap_background_cached(
                pipeline,
                PROCESSED_TRAIN_FILE.stat().st_mtime,
                FINAL_PIPELINE_PATH.stat().st_mtime,
            )
    except Exception:
        pass
    return None


def render_shap_waterfall(input_dict: dict, pipeline):
    """Render a SHAP waterfall plot for the given input."""
    import shap

    st.subheader("Prediction Explainability (SHAP)")
    st.write("This plot shows how each feature contributed to pushing the prediction above or below the baseline.")

    try:
        # Convert input to DataFrame; transform through every pipeline step
        # except the classifier (feature engineering + preprocessing)
        df = pd.DataFrame([input_dict])
        classifier = pipeline.named_steps["classifier"]
        transformer = Pipeline(pipeline.steps[:-1])
        X_trans = np.asarray(transformer.transform(df))
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

        # Create explainer and compute SHAP values
        model_type = type(classifier).__name__
        if model_type in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier"):
            explainer = shap.TreeExplainer(classifier)
        elif hasattr(classifier, "coef_"):
            background = load_shap_background(pipeline)
            if background is None:
                st.info("Training data not found — run the pipeline to enable SHAP for linear models.")
                return
            explainer = shap.LinearExplainer(classifier, background)
        else:
            st.info(f"Local SHAP explanation is not supported for this model type ({model_type}).")
            return

        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
        elif getattr(shap_values, "ndim", 2) == 3:
            shap_values = shap_values[:, :, 1]

        # Create Explanation object
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = expected[1] if len(np.atleast_1d(expected)) > 1 else np.atleast_1d(expected)[0]

        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=expected,
            data=X_trans[0],
            feature_names=feature_names,
        )

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False, max_display=10)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {e}")


# =============================================================================
# UI Layout
# =============================================================================

st.title("📡 Telco Customer Churn Prediction")
st.markdown("""
This application predicts the likelihood of a telecom customer churning (leaving the service).
Use the sidebar to enter customer details manually, or upload a CSV file for batch processing.
""")

# Setup tabs
tab1, tab2, tab3 = st.tabs(["👤 Single Prediction", "📂 Batch Prediction", "📊 Model Insights"])

# --- SIDEBAR: Manual Data Entry ---
st.sidebar.header("Customer Profile")

with st.sidebar.form("prediction_form"):
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    gender = col1.selectbox("Gender", ["Female", "Male"])
    senior = col2.selectbox("Senior Citizen", ["No", "Yes"])
    partner = col1.selectbox("Partner", ["Yes", "No"])
    dependents = col2.selectbox("Dependents", ["No", "Yes"])
    
    st.subheader("Account & Billing")
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=24)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=monthly_charges * tenure)
    
    st.subheader("Services Subscribed")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    
    col3, col4 = st.columns(2)
    security = col3.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = col4.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    protection = col3.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    support = col4.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = col3.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = col4.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    submit_button = st.form_submit_button("Predict Churn Risk", type="primary", use_container_width=True)

# Build input dictionary
input_data = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": protection,
    "TechSupport": support,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}


# --- TAB 1: Single Prediction ---
with tab1:
    if submit_button:
        with st.spinner("Analyzing customer profile..."):
            result = predict_single_api(input_data)
            
            if result:
                col_res1, col_res2 = st.columns([1, 1])
                
                with col_res1:
                    st.subheader("Prediction Result")
                    if result["churn_prediction"] == 1:
                        st.error(f"⚠️ **Customer is Likely to Churn** (Risk: {result['risk_level']})")
                    else:
                        st.success(f"✅ **Customer is Likely to Stay** (Risk: {result['risk_level']})")
                        
                    fig = create_gauge_chart(result["churn_probability"], result["risk_level"])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if result.get("_source") == "local_fallback":
                        st.caption("ℹ️ Computed locally using fallback model (API unreachable).")
                
                with col_res2:
                    pipeline = load_local_model()
                    if pipeline:
                        render_shap_waterfall(input_data, pipeline)
                    else:
                        st.info("Local model not available for SHAP explanation.")
    else:
        st.info("👈 Enter customer details in the sidebar and click **Predict Churn Risk**.")
        
        # Display sample profile
        st.subheader("Sample Customer Profile")
        sample_df = pd.DataFrame([input_data])
        st.dataframe(sample_df, hide_index=True)


# --- TAB 2: Batch Prediction ---
with tab2:
    st.subheader("Batch Prediction from CSV")
    st.write("Upload a CSV file with customer data to generate predictions in bulk.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df_upload)} records. Preview:")
            st.dataframe(df_upload.head(3), hide_index=True)
            
            if st.button("Generate Batch Predictions", type="primary"):
                with st.spinner("Processing batch..."):
                    # Fallback to local prediction for simplicity in Streamlit if API fails
                    pipeline = load_local_model()
                    if pipeline:
                        probs = pipeline.predict_proba(df_upload)[:, 1]
                        threshold = load_decision_threshold()

                        results_df = df_upload.copy()
                        results_df["Churn_Probability"] = np.round(probs, 4)
                        results_df["Churn_Prediction"] = (probs >= threshold).astype(int)
                        results_df["Risk_Level"] = ["High" if p >= 0.65 else ("Medium" if p >= 0.3 else "Low") for p in probs]
                        
                        st.success(f"Successfully processed {len(results_df)} records!")
                        
                        # Display results
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Total Customers", len(results_df))
                        col_b.metric("Predicted Churners", int(results_df["Churn_Prediction"].sum()))
                        col_c.metric("High Risk Count", len(results_df[results_df["Risk_Level"] == "High"]))
                        
                        st.subheader("Results")
                        st.dataframe(
                            results_df.style.map(
                                lambda val: 'color: red' if val == 'High' else ('color: orange' if val == 'Medium' else 'color: green'),
                                subset=['Risk_Level']
                            )
                        )
                        
                        # Download button
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error("Model not available for local prediction.")
        except Exception as e:
            st.error(f"Error processing file: {e}")


# --- TAB 3: Model Insights ---
with tab3:
    st.subheader("Global Model Insights")
    st.write("These visualizations are generated during the model training phase and show overall patterns across the entire dataset.")
    
    try:
        from src.config import FIGURES_DIR
        
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.write("**Feature Importance (SHAP)**")
            shap_bar = FIGURES_DIR / "shap_bar.png"
            if shap_bar.exists():
                st.image(str(shap_bar), use_container_width=True)
            else:
                st.info("SHAP bar plot not found. Run the training pipeline first.")
                
            st.write("**Model Comparison (ROC Curves)**")
            roc_plot = FIGURES_DIR / "roc_curves_comparison.png"
            if roc_plot.exists():
                st.image(str(roc_plot), use_container_width=True)
                
        with col_img2:
            st.write("**Overall Feature Impact (SHAP Summary)**")
            shap_summary = FIGURES_DIR / "shap_summary.png"
            if shap_summary.exists():
                st.image(str(shap_summary), use_container_width=True)
            else:
                st.info("SHAP summary plot not found.")
                
            st.write("**Churn by Contract Type**")
            contract_plot = FIGURES_DIR / "eda_churn_by_contract.png"
            if contract_plot.exists():
                st.image(str(contract_plot), use_container_width=True)
                
    except Exception as e:
        st.error(f"Could not load insights: {e}")
