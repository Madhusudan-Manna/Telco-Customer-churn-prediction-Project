# Data Folder

Place the raw dataset file here before running the notebooks.

## Download

Download the CSV from Kaggle:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Expected file

```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Dataset Summary

| Property | Value |
|---|---|
| Rows | 7,043 |
| Columns | 21 |
| Target | `Churn` (Yes / No) |
| Churn rate | ~26.5% |

## Key Columns

| Column | Type | Notes |
|---|---|---|
| `customerID` | string | Unique identifier — dropped during preprocessing |
| `gender` | categorical | Male / Female |
| `SeniorCitizen` | int | 0 or 1 |
| `tenure` | int | Months with the company |
| `MonthlyCharges` | float | Monthly bill amount |
| `TotalCharges` | string → float | Contains blank strings → coerced to NaN → filled with 0 |
| `Churn` | string → int | Target: Yes→1, No→0 |

> The dataset file is excluded from version control via `.gitignore`.
