# 📦 Customer Churn Prediction

## 🎯 Objective

Predict whether a customer is likely to **churn** (i.e., leave a telecom company) based on their account information, demographics, and usage patterns.

This is a **binary classification** problem.

---

## 📁 Dataset

We used the [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.

It contains 7,043 customer records and 21 features including:

* Customer account info (e.g., `tenure`, `MonthlyCharges`, `Contract`)
* Service usage (e.g., `InternetService`, `StreamingTV`)
* Demographics (e.g., `gender`, `SeniorCitizen`)
* Target variable: `Churn` (Yes/No)

---

## 🧠 Problem Statement

The telecom company wants to:

* Understand **why customers are churning**
* **Predict churn** so they can offer incentives or interventions to retain them

---

## 🔧 Project Tasks

### ✅ 1. Load and Inspect Data

* Used `pandas` to load the dataset
* Inspected:

  * `.shape`, `.info()`, `.describe()`
  * Null values and datatype mismatches
* Handled non-numeric `TotalCharges` entries

### ✅ 2. Exploratory Data Analysis (EDA)

* Analyzed `Churn` distribution (26.5% Yes, 73.5% No)
* Visualized churn by:

  * Contract Type
  * Payment Method
  * Internet Service
* Correlation matrix for numeric features
* Discovered:

  * Higher churn in month-to-month contracts
  * Customers with fiber internet and electronic checks tend to churn more

### ✅ 3. Data Preprocessing

* Converted `TotalCharges` to numeric
* Dropped irrelevant columns (`customerID`)
* Encoded categorical features using `pd.get_dummies`
* Skipped scaling since tree models aren’t sensitive to it

### ✅ 4. Train-Test Split

* Used 80/20 split via `train_test_split`
* Stratified sampling to maintain churn ratio

### ✅ 5. Model Building

We trained two models:

* **Logistic Regression** (baseline)
* **Random Forest Classifier** (tree-based, more powerful)

### ✅ 6. Model Evaluation

We evaluated models on the test set using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

---

## 📊 Evaluation Metrics

### ⚙️ Logistic Regression

| Metric       | Value |
| ------------ | ----- |
| **Accuracy** | 80.3% |
| Precision    | 64.7% |
| Recall       | 57.2% |
| F1 Score     | 60.7% |

**Confusion Matrix**:

```
[[916 117]
 [160 214]]
```

* Performs well overall, with good precision.
* Struggles slightly with recall (misses some churners).
* Highly interpretable model.

---

### 🌲 Random Forest Classifier

| Metric       | Value |
| ------------ | ----- |
| **Accuracy** | 78.9% |
| Precision    | 62.6% |
| Recall       | 51.9% |
| F1 Score     | 56.7% |

**Confusion Matrix**:

```
[[917 116]
 [180 194]]
```

* Performs slightly worse than Logistic Regression.
* Can potentially improve with hyperparameter tuning.
* Captures some non-linear patterns but still underfits a bit.

---

## 🧠 Key Insights

* Month-to-month contract customers are at **higher churn risk**
* Electronic check users churn more than automatic bank/credit users
* Customers with **Fiber Optic Internet** have higher churn
* Longer tenures and bundled services (e.g., multiple lines, streaming) tend to reduce churn

---

## 💾 Next Steps (Optional)

* Tune Random Forest with `GridSearchCV` or `RandomizedSearchCV`
* Try **XGBoost**, **SVM**, or **K-Nearest Neighbors**
* Create a **Streamlit** app for interactive churn prediction

---

## 🛠 Tools Used

* Python 🐍
* Pandas, NumPy
* Seaborn, Matplotlib
* Scikit-learn
* Jupyter Notebook

---

## 📂 File Structure

```
📁 customer-churn-prediction/
│
├── customer_churn_prediction.ipynb   # Final notebook
├── README.md                         # This file
└── churn_data.csv                    # Dataset
```


