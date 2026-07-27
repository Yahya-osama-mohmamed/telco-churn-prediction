# 📡 Telco Customer Churn Prediction

[![CI Pipeline](https://github.com/Yahya-osama-mohmamed/telco-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Yahya-osama-mohmamed/telco-churn-prediction/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning project to predict telecom customer churn. This repository contains a production-ready ML pipeline, a FastAPI REST service, a Streamlit dashboard, and complete deployment configurations.

![Dashboard usage](docs/churn_ui.gif)

*Live dashboard: enter a customer profile → churn probability gauge + per-prediction SHAP explanation.*

![Model metrics](docs/model_metrics.png)


## 🎯 Business Problem

Customer churn (attrition) is one of the most critical challenges for telecom companies. Acquiring a new customer costs **5–25x** more than retaining an existing one. This project builds a predictive system that identifies customers with a high probability of churning, enabling proactive and targeted retention campaigns.

**Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
(7,043 customers, 21 features covering demographics, account info, and services).

---

## 🛠️ Tech Stack

- **Data Science Core:** pandas, NumPy, scikit-learn
- **Machine Learning Models:** XGBoost, LightGBM, Logistic Regression
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **Experiment Tracking:** MLflow
- **API & Backend:** FastAPI, Uvicorn, Pydantic
- **Frontend / Dashboard:** Streamlit, Plotly
- **DevOps / CI-CD:** Docker, Docker Compose, GitHub Actions, Render
- **Testing:** Pytest

---

## 🚀 Quick Start

### 1. Local Setup (Without Docker)

Clone the repository and install dependencies:
```bash
git clone https://github.com/Yahya-osama-mohmamed/telco-churn-prediction.git
cd telco-churn-prediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the ML Pipeline
This will download the dataset, clean data, engineer features, train models, generate SHAP explanations, track experiments in MLflow, and save the final `.joblib` models.
```bash
python main.py
```

### 3. Start the Applications
**FastAPI Backend (Port 8000):**
```bash
uvicorn app.api:app --reload
```
Swagger documentation available at: http://localhost:8000/docs

**Streamlit Dashboard (Port 8501):**
```bash
streamlit run app/streamlit_app.py
```

---

## 🐳 Docker Setup

You can run the entire application stack using Docker Compose. The pipeline (`main.py`) must be run at least once locally to generate the `models/` directory before starting the containers.

```bash
# Start both API and Streamlit containers
docker-compose up --build -d

# Check logs
docker-compose logs -f
```

---

## 📂 Project Structure

```
.
├── app/                    # Deployment Layer
│   ├── api.py              # FastAPI application
│   ├── schemas.py          # Pydantic models
│   └── streamlit_app.py    # Streamlit dashboard
├── data/                   # Data directory (ignored in git)
│   ├── raw/                # Original downloaded dataset
│   └── processed/          # Cleaned & split data
├── figures/                # EDA and SHAP visualizations
├── models/                 # Saved joblib models and pipelines
├── mlruns/                 # MLflow tracking store
├── notebooks/              # Jupyter notebooks for exploration
├── reports/                # Executive summaries and comparison tables
├── src/                    # Core ML Source Code
│   ├── config.py           # Constants and paths
│   ├── data_loader.py      # Download and initial profiling
│   ├── preprocessing.py    # Cleaning, encoding, and scaling pipelines
│   ├── feature_engineering.py # Feature creation
│   ├── feature_selection.py   # MI and Random Forest importance
│   ├── model_training.py   # RandomizedSearchCV tuning
│   ├── model_evaluation.py # Metrics computation and plotting
│   ├── explainability.py   # SHAP analysis
│   └── logger.py           # Structured JSON logging
├── tests/                  # Pytest unit tests
├── Dockerfile              # Multi-purpose Dockerfile
├── docker-compose.yml      # Container orchestration
├── main.py                 # Pipeline execution entry point
├── render.yaml             # Render cloud deployment config
└── requirements.txt        # Python dependencies
```

---

## 📊 Model Performance

After running the pipeline, check `reports/model_comparison.csv` for detailed metrics.

Our best model (typically **XGBoost** or **LightGBM**) achieves:
- **ROC-AUC:** ~0.84 - 0.85
- **Recall (Churn):** Prioritized via `scale_pos_weight` to identify as many at-risk customers as possible.

### Methodology Notes

- **Model selection** uses the validation set (15%); the test set (15%) is reserved
  strictly for the final unbiased performance estimate.
- **Feature engineering lives inside the sklearn Pipeline** (`FeatureEngineer` step),
  so statistics such as the MonthlyCharges median used by `high_value_short_tenure`
  are learned from training folds only. The saved `models/final_pipeline.joblib`
  accepts raw customer records — no manual feature engineering is needed at serving time.
- **The decision threshold** is tuned on the validation set (maximizing F1) and stored
  in `models/model_metadata.json`; the API applies it automatically instead of a
  hardcoded 0.5.
- Feature-importance analysis (mutual information + random forest) is reported in
  `figures/feature_selection_*.png`; models are trained on the full feature set.

### Feature Importance (SHAP)
Top drivers of churn identified by the model:
1. **Contract Type:** Month-to-month contracts have vastly higher churn rates.
2. **Tenure:** Shorter tenure indicates higher risk.
3. **Internet Service:** Fiber optic customers show unexpectedly high churn (potential service quality issue).
4. **Total / Monthly Charges:** Higher charges correlate with higher churn.

---

## ☁️ Deployment

The project is configured for seamless deployment on **Render**.

1. Connect your GitHub repository to Render.
2. The `render.yaml` Blueprint automatically provisions:
   - A Web Service for the FastAPI backend.
   - A Web Service for the Streamlit dashboard.
3. CI is handled automatically via GitHub Actions (`.github/workflows/ci.yml`), which runs all `pytest` suites before deployment.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🖼️ Output Gallery

| | |
|---|---|
| ![Churn drivers](figures/shap_bar.png) | ![SHAP beeswarm](figures/shap_summary.png) |
| ![ROC curves](figures/roc_curves_comparison.png) | ![PR curves](figures/pr_curves_comparison.png) |
| ![Model comparison](figures/model_comparison_bar.png) | ![Churn by contract](figures/eda_churn_by_contract.png) |
| ![Feature selection MI](figures/feature_selection_mi.png) | ![Confusion matrix](figures/confusion_matrix_logistic_regression_test.png) |
