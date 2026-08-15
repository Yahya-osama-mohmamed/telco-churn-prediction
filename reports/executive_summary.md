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

**Evaluation Strategy:** Models were tuned with 5-fold stratified cross-validation on the training split (randomised search, up to 50 candidates, scored on **ROC-AUC**), selected on a held-out 15% validation set, and reported once on an unseen 15% test set. Class imbalance was handled natively using algorithm-specific weights (`scale_pos_weight` / balanced class weights). The production decision threshold is tuned on the validation set rather than fixed at 0.5 — it maximises **F1**, which balances precision and recall rather than maximising recall alone.

### Final Model Selection
The champion is chosen automatically by validation ROC-AUC each time the pipeline runs; all three candidates score within ~0.002 AUC of each other on this dataset, and the current champion is recorded in `models/model_metadata.json` alongside its tuned decision threshold and test metrics. In the latest run **Logistic Regression** narrowly won (test ROC-AUC ≈ 0.85) — a welcome outcome operationally, as it is the fastest and most interpretable of the three.
* At the tuned threshold of 0.669 the model recovers **62.1% of churners at 61.3% precision** (accuracy 0.796, F1 0.617) on the held-out test set. Maximising F1 deliberately buys precision at the cost of recall: at the default 0.5 cut-off the same model reaches 78.2% recall but precision falls to 50.8%, so nearly half of every contacted customer would be a false alarm. Which operating point is correct depends on the retention offer's cost, and that trade-off is the decision the business owns — see `models/model_metadata.json` for the shipped point.
* The train vs. validation gap is minimal, indicating no significant overfitting.

## 4. Drivers of Churn (SHAP Explainability)
SHAP values explain the **shipped Logistic Regression pipeline** — the same artifact the API serves, not a separate tree model — so the drivers below are the reasons behind the scores stakeholders actually receive. The top 5 drivers globally are:
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
