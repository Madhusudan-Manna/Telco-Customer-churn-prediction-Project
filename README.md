# Telco Customer Churn Prediction

**Internship Project — E&ICT Academy, IIT Guwahati**  
**Student :** Madhusudan Manna | **Batch :** Batch-13

---

## What is this project about?

A telecom company wants to know which customers are likely to **leave** (called "churn").
If we can predict churn in advance, the company can take steps to retain those customers
and reduce revenue loss.

In this project I built and compared **7 machine learning classification models** to
predict customer churn using the IBM Telco Customer Churn dataset.

---

## Dataset

| Property | Value |
|---|---|
| Source | IBM Sample Dataset (Telco Customer Churn) |
| File | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Rows | 7,043 customers |
| Columns | 21 (demographics, services, billing, churn label) |

> **Note:** The dataset file is not included in the repository (added to `.gitignore`).
> Download it from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
> and place it inside the `data/` folder.

---

## Project Structure

```
2.Telco Customer Churn/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← raw dataset (download separately)
│   └── README.md
│
├── notebook/
│   ├── 01_data_exploration.ipynb               ← data loading, shape, nulls, dtypes
│   ├── 02_eda.ipynb                            ← EDA, visualisations, skewness
│   └── 03_preprocessing.ipynb                 ← preprocessing, feature engineering, scaling
│
├── src/
│   ├── __init__.py
│   └── preprocessing.py                        ← reusable preprocessing functions
│
├── models/
│   ├── best_model.pkl                          ← saved best model (created on run)
│   ├── scaler.pkl                              ← saved scaler  (created on run)
│   └── README.md
│
├── reports/
│   ├── churn_rate_by_contract.png
│   ├── correlation_heatmap.png
│   ├── monthly_charges_by_churn.png
│   └── README.md
│
├── images/                                     ← folder for additional plots
├── .gitignore
├── requirements.txt
└── README.md                                   ← this file
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Load dataset, inspect shape, dtypes, nulls, duplicates, churn balance |
| `02_eda.ipynb` | Exploratory data analysis — distributions, correlations, churn by category |
| `03_preprocessing.ipynb` | Clean data, feature engineering (tenure_group), encode + scale, pipeline |

---

## Tasks Completed

| Task | What was done |
|------|---------------|
| **Task 1** | Imported all required libraries with comments |
| **Task 2** | Loaded dataset; fixed TotalCharges dtype; handled NaN with fillna(0) |
| **Task 3** | Explored dataset — shape, dtypes, statistics |
| **Task 4** | Cleaned data — dropped customerID, encoded Churn as 0/1 |
| **Task 5** | EDA — churn distribution, correlation heatmap, feature distributions, skewness |
| **Task 6** | Feature engineering — tenure_group, train/test split (80/20 stratified) |
| **Task 7** | Encoding with pd.get_dummies + StandardScaler (fixed ValueError on string columns) |
| **Task 8** | ColumnTransformer pipeline (StandardScaler + OneHotEncoder) |

---

## Models Used

| Model | Type |
|---|---|
| Logistic Regression | Linear |
| Decision Tree | Tree-based |
| Random Forest | Ensemble (Bagging) |
| Gradient Boosting | Ensemble (Boosting) |
| SVM | Kernel-based |
| KNN | Distance-based |
| Naive Bayes | Probabilistic |

---

## How to Run

**Step 1 — Clone the repository**
```bash
git clone https://github.com/<your-username>/telco-customer-churn.git
cd "2.Telco Customer Churn"
```

**Step 2 — Create and activate a virtual environment**
```bash
python -m venv myenv
myenv\Scripts\activate      # Windows
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
and place the CSV file in the `data/` folder:
```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

**Step 5 — Open and run the notebooks in order**
```bash
jupyter notebook
```
Run notebooks in this order:
1. `01_data_exploration.ipynb`
2. `02_eda.ipynb`
3. `03_preprocessing.ipynb`

---

## Key Findings

- About **26%** of customers have churned
- **Tenure**, **TotalCharges**, and **MonthlyCharges** are the most important features
- Customers on **Month-to-month contracts** are most likely to churn
- **Gradient Boosting** and **Random Forest** achieved the best ROC-AUC scores

---

## Requirements

All packages listed in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| pandas | Data loading and manipulation |
| numpy | Numerical calculations |
| matplotlib | Basic plotting |
| seaborn | Statistical visualisation |
| scikit-learn | ML models and evaluation metrics |
| jupyter | To run the notebooks |

---

*This project was completed as part of the E&ICT Academy, IIT Guwahati AI/ML/DS Internship.*
