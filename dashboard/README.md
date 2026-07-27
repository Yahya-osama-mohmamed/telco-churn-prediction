# Customer Retention Command Center — Power BI Dashboard

## How to open

1. Double-click `ChurnRetention/ChurnRetention.pbip` (opens in Power BI Desktop).
2. On first open, click **Refresh now** in the yellow banner (or Home → Refresh).
   PBIP projects store the model definition as text; the data itself loads from
   the CSVs in `data/` on first refresh (~5 seconds).
3. Save once after refreshing — Power BI caches the data locally after that.

## Pages & the story they tell

| Page | Business question |
|---|---|
| **Executive Overview** | How big is the churn problem and what revenue is exposed? KPIs, risk mix, and the three structural drivers (contract, tenure, payment method). |
| **Risk Segmentation** | Where does churn concentrate? Slice by risk level, contract, internet, payment; customer-level scatter and the contract × internet hot-spot matrix. |
| **Retention Targeting** | Who do we call first? A call list ranked by revenue-at-risk, with savable-revenue scenarios (30% win-back assumption). |
| **Model Insights** | Why trust the scores? Champion model card, SHAP drivers, candidate comparison, and the calibration chart (actual churn rises with predicted probability). |

Every page has slicers; all visuals cross-filter each other (click any bar/segment
to filter the page).

## Data

All data in `data/*.csv` is **model-generated**: every one of the 7,043 customers
was scored by the trained pipeline (`models/final_pipeline.joblib`), including
churn probability, risk tier, and monthly revenue at risk. Regenerate after a
retrain by re-running the scoring export.

The semantic model loads the CSVs by absolute path. If you move the project,
update the paths: open Transform Data in Power BI Desktop, or edit the
`File.Contents("...")` paths in `ChurnRetention.SemanticModel/model.bim`.
