# Executive Summary: Telco Customer Churn Prediction

## 1. Business Context & Objective
Customer churn is a critical profitability leak for telecom providers, with acquisition costs far exceeding retention costs. The objective of this project was to develop a machine learning system capable of predicting which customers are at high risk of churning, allowing the business to proactively target them with retention campaigns (e.g., discounts, contract upgrades).

## 2. Key Findings from Data Exploration (EDA)
Analysis of 7,043 customer records revealed several critical insights into churn behavior:
* **Overall Churn Rate:** 26.5% of the customer base churned in the last month.
* **Contract Vulnerability:** Customers on **Month-to-month contracts** are highly volatile, churning at a rate of 42.7%, compared to just 11.2% for One-year and 2.8% for Two-year contracts.
* **The "Fiber Optic" Problem:** Customers with Fiber Optic internet churn at roughly double the rate of DSL customers. This suggests a potential issue with service quality, pricing, or competitor aggression in the fiber market.
* **Tenure Loyalty:** The highest risk period is the first 12 months. If a customer can be retained past the 2-year mark, their churn probability drops significantly.
* **Electronic Check Friction:** Customers paying via Electronic Check have disproportionately high churn compared to automated methods (Credit Card / Bank Transfer).

## 3. Model Development & Performance
We developed a complete machine learning pipeline incorporating data cleaning, custom feature engineering, and robust feature selection. We trained three algorithms: Logistic Regression (interpretable baseline), XGBoost, and LightGBM.

**Evaluation Strategy:** Models were tuned using 5-fold cross-validation, selected on a held-out 15% validation set, and reported on an unseen 15% test set, prioritizing **ROC-AUC** and **Recall** (to ensure we capture as many true churners as possible). Class imbalance was handled natively using algorithm-specific weights (`scale_pos_weight` / balanced class weights). The production decision threshold is tuned on the validation set rather than fixed at 0.5.

### Final Model Selection
The champion is chosen automatically by validation ROC-AUC each time the pipeline runs; all three candidates score within ~0.002 AUC of each other on this dataset, and the current champion is recorded in `models/model_metadata.json` alongside its tuned decision threshold and test metrics. In the latest run **Logistic Regression** narrowly won (test ROC-AUC ≈ 0.85) — a welcome outcome operationally, as it is the fastest and most interpretable of the three.
* It successfully identifies the vast majority of at-risk customers (~78% recall at the tuned threshold).
* The train vs. validation gap is minimal, indicating no significant overfitting.

## 4. Drivers of Churn (SHAP Explainability)
Using Game Theory (SHAP values), we cracked open the "black box" of the XGBoost model to understand exactly *why* it predicts churn. The top 5 drivers globally are:
1. **Contract (Month-to-month):** The single biggest risk factor.
2. **Tenure:** Shorter tenure increases risk significantly.
3. **Internet Service (Fiber Optic):** Strongly pushes the model toward predicting churn.
4. **Total Charges / Monthly Charges:** High monthly spend relative to tenure is a strong churn signal.
5. **Payment Method (Electronic Check):** Increases risk.

Conversely, having multiple services (Phone + Internet + Security) acts as a strong anchor, reducing churn probability due to high switching costs.

## 5. Business Recommendations
1. **Incentivize Contract Upgrades:** The highest ROI action is migrating Month-to-month customers to One-year contracts. Offer targeted discounts on month 10-12 to lock them in.
2. **Investigate Fiber Optic Service:** The high churn rate in the premium Fiber segment requires immediate operational review. Are there outages? Are competitors undercutting price?
3. **Promote Automated Payments:** Offer a small monthly discount ($2-$5) for customers who switch from Electronic Check to Auto-pay via Credit Card or Bank Transfer.
4. **Bundle "Sticky" Services:** Customers with Tech Support and Online Security churn less. Offer these as free 3-month trials to new customers to increase switching friction.

## 6. Estimated ROI
Implementing this model allows the retention team to transition from "spray and pray" marketing to targeted interventions. By focusing retention budgets only on the top 20% of customers identified as "High Risk" by the model, the company can significantly reduce marketing spend while preventing high-value customer attrition.
