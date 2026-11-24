# Electricity Theft & Abnormal Consumption Analytics

Streamlit prototype for electricity theft detection and abnormal smart-meter consumption analytics.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
Upload a labelled CSV with one consumer per row, a binary target column named `FLAG` (0=normal, 1=labelled suspicious), and numeric reading columns. You can change the target column in the sidebar. Exclude ID columns such as `CONS_NO`.

## Features
- Consumer-level statistical feature engineering
- Logistic Regression, Random Forest and HistGradientBoosting baselines
- Isolation Forest anomaly score
- Configurable hybrid score and ranked review candidates
- Confusion matrix, metrics and CSV export

**Important:** Synthetic demo data is illustrative only. A flagged pattern is not proof of theft. Validate using the actual permitted dataset, leakage-safe chronological/customer-grouped splits, and human review. Do not publish confidential customer data.