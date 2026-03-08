# 📉 Customer Churn Prediction — Neo Telecom

### End-to-End Binary Classification · EDA · Multi-Model Comparison · Business Intelligence

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

-----

## 📌 Business Problem

**Neo Telecom** is losing customers to competitors. Every lost customer is lost revenue — and acquiring a new customer costs 5–7× more than retaining an existing one.

**This project answers three questions:**

1. **Who** is most likely to churn? (Customer segmentation & EDA)
1. **Can we predict** whether a customer will churn — before they do? (ML Classification)
1. **What should Neo do?** (Actionable business recommendations)

> **Key result:** The model correctly identifies the majority of at-risk customers, enabling Neo’s retention team to act proactively — estimated to protect **₹3.3Cr+ in annual revenue**.

-----

## 📁 Project Structure

```
customer-churn-prediction/
│
├── Customer_Churn_UPGRADED.ipynb   ← Main analysis notebook (fully commented)
├── customer churn.csv              ← Dataset (7,043 customers × 21 features)
├── churn_model.pkl                 ← Saved best model (joblib)
├── churn_scaler.pkl                ← Saved StandardScaler (joblib)
└── README.md                       ← This file
```

-----

## 📊 Dataset Overview

|Property              |Details                                                |
|----------------------|-------------------------------------------------------|
|**Source**            |Neo Telecom customer database                          |
|**Rows**              |7,043 customers                                        |
|**Columns**           |21 (20 features + 1 target)                            |
|**Target**            |`Churn` — Yes (customer left) / No (customer stayed)   |
|**Class Distribution**|No: 73.5% (5,174) · Yes: 26.5% (1,869) — **Imbalanced**|

### Feature Categories

|Category        |Features                                                                                                                               |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------|
|**Demographics**|gender, SeniorCitizen, Partner, Dependents                                                                                             |
|**Services**    |PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies|
|**Account**     |tenure, Contract, PaperlessBilling, PaymentMethod                                                                                      |
|**Billing**     |MonthlyCharges, TotalCharges                                                                                                           |

-----

## 🔬 What This Project Covers (Step by Step)

### STEP 1 — Data Cleaning

- Detected and fixed a **hidden data bug**: `TotalCharges` was stored as `object` (string) due to blank spaces in 11 rows
- Used `pd.to_numeric(errors='coerce')` to convert, filled nulls with **median** (robust to outliers)
- Dropped `customerID` — an identifier with zero predictive power
- Checked for duplicate rows and null values across all columns

### STEP 2 — Advanced EDA

Every chart answers a specific business question:

**Target Distribution**

- Only 26.5% of customers churned — dataset is imbalanced
- A naive model predicting “No Churn” always would get 73.5% accuracy but catch **zero churners** — this is why accuracy alone is misleading

**Categorical Feature Analysis**

- Churn rate by: Contract type, PaymentMethod, InternetService, SeniorCitizen, Partner, Dependents, PaperlessBilling, gender
- **Key finding**: Month-to-month contract customers churn at **~43%** vs only **~11%** for two-year contracts

**Numeric Feature Analysis**

- KDE plots + boxplots for tenure, MonthlyCharges, TotalCharges split by Churn
- **Key finding**: Churned customers have median tenure ~10 months vs ~38 months for loyal customers — **the first year is the highest-risk window**

**Correlation Heatmap**

- tenure vs TotalCharges: strong positive correlation (~0.83) — expected, longer customers pay more over time
- Churn vs tenure: negative — longer tenure = less likely to churn
- Churn vs MonthlyCharges: positive — higher charges = higher churn risk

### STEP 3 — Feature Engineering & Preprocessing

|Step                |What & Why                                                                                                      |
|--------------------|----------------------------------------------------------------------------------------------------------------|
|**Target Encoding** |Churn: Yes→1, No→0                                                                                              |
|**LabelEncoder**    |Binary columns (gender, Partner, Dependents, PhoneService, PaperlessBilling, MultipleLines) — only 2 values each|
|**One-Hot Encoding**|Multi-category columns (Contract, InternetService, PaymentMethod, etc.) — avoids false ordinal relationships    |
|**Stratified Split**|80:20 train-test split with `stratify=y` — preserves 26.5% churn ratio in both sets                             |
|**StandardScaler**  |Fit on train only, transform both — **prevents data leakage**                                                   |


> ⚠️ **Why LabelEncoder for binary but One-Hot for multi-category?**  
> LabelEncoding `Contract` as `Month-to-month=0, One year=1, Two year=2` implies a false mathematical ordering. One-hot encoding creates separate binary columns — no ordering implied.

### STEP 4 — Model Building (4 Models Compared)

All models trained with `class_weight='balanced'` to handle the 73/27 imbalance.

|Model                  |Why Used                                                         |
|-----------------------|-----------------------------------------------------------------|
|**Logistic Regression**|Linear baseline; interpretable coefficients                      |
|**Decision Tree**      |Non-linear baseline; visualisable rules                          |
|**Random Forest**      |Ensemble of 100 trees; reduces variance; gives feature importance|
|**Gradient Boosting**  |Sequential error-correcting; high accuracy on tabular data       |

### STEP 5 — Evaluation Strategy

**Primary metric: ROC-AUC** (not accuracy)

> Accuracy is misleading on imbalanced data. ROC-AUC measures how well the model ranks churners above non-churners across all thresholds. AUC=0.5 → random guessing. AUC=1.0 → perfect model.

- **ROC Curves** — all 4 models on one chart
- **Confusion Matrices** — TP, TN, FP, FN for all models
- **5-Fold StratifiedKFold Cross-Validation** — mean AUC ± std dev (more reliable than single split)
- **Classification Report** — Precision, Recall, F1 per class for the best model

> For churn, **Recall on the Churned class** matters most. Missing a churner (False Negative) = no retention action = lost customer. A wrongly flagged non-churner (False Positive) = we offer them a discount = small cost.

### STEP 6 — Feature Importance

Random Forest `feature_importances_` — top 15 features ranked by average Gini impurity reduction:

|Rank|Feature            |Business Meaning              |
|----|-------------------|------------------------------|
|1   |`tenure`           |New customers churn most      |
|2   |`MonthlyCharges`   |Price sensitivity drives churn|
|3   |`TotalCharges`     |Proxy for long-term value     |
|4   |`Contract_Two year`|Long contracts = loyalty      |
|5   |`TechSupport_Yes`  |Support reduces churn         |

### STEP 7 — Model Saving

```python
import joblib

# Save
joblib.dump(best_model, 'churn_model.pkl')
joblib.dump(scaler,     'churn_scaler.pkl')

# Load & predict on new customers
loaded_model  = joblib.load('churn_model.pkl')
loaded_scaler = joblib.load('churn_scaler.pkl')

new_data_scaled = loaded_scaler.transform(new_customer_df)
prediction      = loaded_model.predict(new_data_scaled)
probability     = loaded_model.predict_proba(new_data_scaled)[:, 1]

print("Churn prediction:", "Will Churn" if prediction[0] == 1 else "Will Stay")
print("Churn probability:", f"{probability[0]*100:.1f}%")
```

-----

## 📈 Results

|Model              |Accuracy|ROC-AUC|CV Mean AUC|
|-------------------|--------|-------|-----------|
|Logistic Regression|~80%    |~84%   |~83% ± 1%  |
|Decision Tree      |~78%    |~79%   |~77% ± 2%  |
|Random Forest      |~81%    |~85%   |~84% ± 1%  |
|Gradient Boosting  |~82%    |~86%   |~85% ± 1%  |


> Best model selected by highest cross-validation ROC-AUC.

-----

## 💡 Business Recommendations (for Neo Telecom)

|#|Finding                                                  |Recommended Action                                                         |
|-|---------------------------------------------------------|---------------------------------------------------------------------------|
|1|New customers churn within first 12 months               |Launch 12-month loyalty program — discounts at month 3, 6, and 12          |
|2|Month-to-month contracts = 3× higher churn than 2-year   |Offer 1 free month to any customer who upgrades to an annual plan          |
|3|High MonthlyCharges customers are price sensitive        |Proactive retention call + loyalty discount for customers paying >$80/month|
|4|Senior citizens churn at higher rates                    |Dedicated senior helpline + simplified plan options                        |
|5|Customers without TechSupport & OnlineSecurity churn more|Bundle these as free add-ons for all customers in their first 6 months     |

### Estimated Business Impact

```
Model correctly identifies ~60% of the 1,869 at-risk customers
If Neo retains 30% of correctly flagged churners = ~336 customers saved
At average ₹500/month per customer = ₹3,36,00,000 (₹3.36 Cr) annual revenue protected
```

-----

## 🛠️ Tech Stack

|Tool                    |Purpose                                           |
|------------------------|--------------------------------------------------|
|**Python 3.10**         |Core language                                     |
|**Pandas**              |Data loading, cleaning, EDA                       |
|**NumPy**               |Numeric operations                                |
|**Matplotlib / Seaborn**|Visualisations — KDE, boxplot, heatmap, bar charts|
|**Scikit-learn**        |Preprocessing, models, evaluation metrics         |
|**joblib**              |Model serialisation                               |
|**Google Colab**        |Development environment                           |

-----

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

1. Open `Customer_Churn_UPGRADED.ipynb` in [Google Colab](https://colab.research.google.com/)
1. Upload `customer churn.csv` to the Colab files panel
1. Run all cells (`Runtime → Run all`)

### Option 2 — Local (Jupyter)

```bash
# 1. Clone the repo
git clone https://github.com/sairamsirikonda/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn joblib

# 3. Launch Jupyter
jupyter notebook Customer_Churn_UPGRADED.ipynb
```

> Make sure `customer churn.csv` is in the same folder as the notebook.

-----

## 🧠 Key Learning Points

|Concept                 |How It Appears in This Project                                             |
|------------------------|---------------------------------------------------------------------------|
|**Class Imbalance**     |73.5% vs 26.5% — handled with `class_weight='balanced'` and StratifiedKFold|
|**Data Leakage**        |Scaler fit on train only — never on test or full data                      |
|**Metric Selection**    |ROC-AUC chosen over accuracy for imbalanced target                         |
|**Encoding Strategy**   |LabelEncoder for binary; One-Hot for multi-category nominal                |
|**Cross-Validation**    |5-Fold Stratified CV for reliable, stable performance estimates            |
|**Feature Importance**  |Random Forest importance scores → direct business actions                  |
|**Production Readiness**|Model + scaler saved as `.pkl` with load-and-predict snippet               |

-----

## 👤 Author

**Sairam Sirikonda**  
Data Scientist · Machine Learning Engineer · Computer Vision  
📧 sairam.7751@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/sairamsirikonda) · [GitHub](https://github.com/sairamsirikonda)

-----

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
