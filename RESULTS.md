# RESULTS — Telco Churn Prediction

Engineering source of truth. Every number here is reproduced from a file in this
repository; the provenance column says which. Business narrative lives in
[`reports/executive_summary.md`](reports/executive_summary.md) — where the two
disagree, this file wins.

Last verified against artifacts trained `2026-08-01T20:35:22Z`
(`models/model_metadata.json`).

---

## 1. Problem

Predict which of an IBM Telco snapshot's 7,043 customers will churn, and ship the
scores behind an API so a retention team can act on them. Binary classification,
26.5% positive rate, static single-snapshot data.

## 2. Protocol

| Decision | Value | Where |
|---|---|---|
| Split | 70 / 15 / 15 stratified, before any fitting | `data/processed/{train,validation,test}.csv` |
| Split sizes | 4,932 / 1,054 / 1,057 | row counts of the above |
| Feature engineering | inside the sklearn `Pipeline` (`FeatureEngineer`) | `pipeline_lib.py` |
| Tuning | `RandomizedSearchCV`, `n_iter=min(50, grid)`, `StratifiedKFold(5, shuffle)`, `scoring="roc_auc"` | notebook cell 27 |
| Imbalance | `class_weight="balanced"` / `scale_pos_weight` — no resampling | notebook |
| Champion selection | highest **validation** ROC-AUC | notebook |
| Threshold | argmax F1 on the **validation** PR curve | notebook cell 35 |
| Test set | opened once, after both choices above were frozen | notebook |

Putting `FeatureEngineer` in the pipeline is the load-bearing leakage control:
the `high_value_short_tenure` flag depends on a `MonthlyCharges` median, which is
therefore learned per training fold rather than from the whole dataset.

## 3. Model comparison

Source: [`reports/model_comparison.csv`](reports/model_comparison.csv). Threshold-free
columns only — the point metrics in that file are computed at 0.5, **not** at the
shipped threshold (see §4).

| Model | Val ROC-AUC | Val PR-AUC | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---|
| **Logistic Regression** (champion) | **0.8367** | 0.6400 | 0.8538 | 0.6736 |
| XGBoost | 0.8356 | 0.6401 | 0.8555 | 0.6700 |
| LightGBM | 0.8339 | 0.6337 | 0.8548 | 0.6671 |

Spread across all three on validation: **0.0028 ROC-AUC**. On test the ordering
flips — XGBoost is nominally best (0.8555 vs 0.8538) — which is the clearest
available evidence that these differences are noise, not signal.

Logistic regression ships. Not because it measured better, but because at a
0.003 tie it is the fastest (6.9 s vs 30.6 s to fit) and the only one whose
coefficients a retention manager can read directly.

## 4. Shipped operating point

Source: [`models/model_metadata.json`](models/model_metadata.json). The API reads
this file at startup and falls back to 0.5 with a logged warning if it cannot
parse it (`app/api.py:98`).

| | Threshold 0.669 (shipped) | Threshold 0.5 (default) |
|---|---|---|
| Accuracy | 0.7956 | 0.7417 |
| Precision | 0.6127 | 0.5081 |
| Recall | 0.6214 | 0.7821 |
| F1 | 0.6170 | 0.6160 |

**The F1 of the two operating points is identical to three decimal places
(0.617 vs 0.616).** F1 was the selection criterion, and it cannot distinguish
them — the curve is flat across this whole region. What actually changes between
the two columns is who bears the error: the shipped point contacts fewer
customers and wastes less budget, the default point catches 16 more percentage
points of churners and wastes more.

Neither is justified by evidence in this repo, because no cost was ever attached
to a retention offer or a missed churner. Choosing on F1 was a proxy for a
business decision nobody made.

## 5. Drivers

SHAP over the shipped logistic pipeline (`figures/shap_bar.png`,
`figures/shap_summary.png`): month-to-month contract, short tenure, fiber-optic
internet, charges, electronic-check payment. Direction and ordering agree with
the unmodelled EDA rates in `reports/executive_summary.md` §2, which is a
consistency check, not independent confirmation.

## 6. What the repository does NOT establish

Listed because the README and dashboard are more confident than the evidence.

1. **No uncertainty on any metric.** No bootstrap CI, no DeLong test, no repeated
   splits — verified absent from the notebook. The 0.0028 champion margin is
   reported against a 1,054-row validation set with no interval around it, so the
   claim "logistic regression won" is not statistically supported. It is a tie
   broken on operational grounds, and §3 states it that way.
2. **Calibration is shown but never measured.** The Power BI *Model Insights*
   page plots observed churn rate per predicted-probability bucket — a reliability
   diagram in effect. No Brier score, no ECE, no calibration curve exists in code,
   and no recalibration is applied. "Stakeholders can trust the scores" is not
   yet earned.
3. **The dashboard scores its own training data.** `dashboard/data/customer_scores.csv`
   holds all 7,043 customers scored by a pipeline fit on 4,932 of them, so 70% of
   every dashboard KPI, the risk-tier mix, the targeting list and the calibration
   chart are in-sample and optimistic. This is documented in `dashboard/README.md`
   but never flagged as a caveat on the pages themselves.
4. **The dashboard cannot be regenerated from this repo.** `dashboard/README.md`
   says to "re-run the scoring export"; no such script exists — the notebook has
   zero references to `dashboard/`. The CSVs are un-versioned outputs of a process
   that was not committed.
5. **The dashboard reports a different operating point than the API.**
   `dashboard/data/model_comparison.csv` carries precision 0.5081 / recall 0.7821
   — the 0.5-threshold numbers — while the API serves 0.669.
6. **No temporal validation.** The data is one snapshot with no timestamps, so
   nothing here speaks to drift, and no drift monitoring exists.
7. **Not deployed.** `render.yaml` is a valid Blueprint; there is no evidence in
   this repository that it has ever run. Docker Compose and CI (pytest, 17 tests,
   plus an API import check) do run.

## 7. What changed because of the evidence

- Threshold moved off 0.5 and into `model_metadata.json` once it was clear the
  default was not defensible — though §4 shows the replacement is not defensible
  either, for a different reason.
- Feature engineering moved from a notebook cell into a pipeline step after the
  median-leak was identified; that is why `pipeline_lib.py` still exists as a
  module (a transformer pickled from `__main__` cannot be loaded by the API).
- Boosted models were kept in the repo rather than deleted, because the flipped
  test-set ordering is the evidence for the tie and is worth showing.

## 8. Next work, in priority order

1. Bootstrap CIs on test ROC-AUC and a DeLong test between the three models —
   converts §3's hand-waving into a measured claim.
2. A cost matrix for retention offers, then re-derive the threshold as expected
   cost rather than F1.
3. Brier score + reliability curve in code; recalibrate if needed.
4. Commit the scoring export, and restrict the dashboard to held-out rows or
   label the in-sample portion.
