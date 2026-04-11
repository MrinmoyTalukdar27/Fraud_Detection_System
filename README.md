# 💳 Fraud Detection System
### Real-Time Financial Fraud Detection using Machine Learning & Streamlit

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

---

## 🚨 What is This?

A complete **end-to-end Machine Learning project** that detects fraudulent financial transactions in real-time. Built on a dataset of **6.3 million transactions**, the system uses a Logistic Regression pipeline and is deployed as a live web application using Streamlit.

Enter transaction details → Click Analyze → Know instantly if it's **FRAUD** or **SAFE** ✅

## 📊 Model Performance

| Metric | Score |
|---|---|
| ✅ Overall Accuracy | **94.63%** |
| 🎯 Fraud Recall | **94%** (catches 94 out of every 100 fraud cases) |
| 📦 Training Data | 6,362,620 transactions |
| 🚨 Fraud Cases Found | 8,213 out of 6.3M |

---

## 🗂️ Project Structure

```
Fraud_Detection_System/
│
├── main.ipynb                      # Jupyter Notebook — Data Analysis & Model Training
├── Fraud_Detection.py              # Streamlit Web Application
├── fraud_detection_pipeline.pkl    # Saved ML Model (generated after running notebook)
├── AIML Dataset.csv                # Dataset (not included — see below)
├── screenshots/
│   ├── screenshot1.png
│   └── screenshot2.png
└── README.md
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| `Python 3.x` | Core programming language |
| `Pandas & NumPy` | Data loading and manipulation |
| `Matplotlib & Seaborn` | Data visualization and EDA |
| `Scikit-Learn` | ML Pipeline, StandardScaler, OneHotEncoder, LogisticRegression |
| `Joblib` | Saving and loading the trained model |
| `Streamlit` | Web application deployment |
| `Anaconda` | Environment management |

---

## 🧠 How It Works

### Step 1 — Data Exploration (EDA)
- Loaded 6.3 million financial transactions
- Found **zero missing values**
- Discovered only **0.13%** of transactions are fraudulent (class imbalance problem)
- Key finding: **Fraud only occurs in TRANSFER and CASH_OUT** transactions

### Step 2 — Feature Engineering
Created two new calculated features to help the model:
```python
df['Balance-Dif-Org']  = df['oldbalanceOrg']  - df['newbalanceOrig']
df['Balance-Dif-Dest'] = df['oldbalanceDest'] - df['newbalanceDest']
```

### Step 3 — ML Pipeline
```python
Pipeline([
    ('prep', ColumnTransformer([
        ('num',  StandardScaler(),          numerical_columns),
        ('cata', OneHotEncoder(drop='first'), ['type'])
    ])),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])
```

- `StandardScaler` — normalises numerical values
- `OneHotEncoder` — converts transaction types to numbers
- `class_weight='balanced'` — fixes the class imbalance problem

### Step 4 — Save & Deploy
```python
joblib.dump(pipeline, 'fraud_detection_pipeline.pkl')  # Save
# Then run the Streamlit app to use it live
```

---

## 🚀 How to Run This Project

### Prerequisites
Make sure you have Anaconda installed. Then:

```bash
# Step 1 — Clone the repository
git clone https://github.com/yourusername/Fraud_Detection_System.git
cd Fraud_Detection_System
```

```bash
# Step 2 — Create and activate environment
conda create -n fraud_env python=3.10
conda activate fraud_env
```

```bash
# Step 3 — Install dependencies
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
```

```bash
# Step 4 — Train the model (run the notebook first)
jupyter notebook main.ipynb
# Run all cells — this generates fraud_detection_pipeline.pkl
```

```bash
# Step 5 — Launch the web app
streamlit run Fraud_Detection.py
```

Then open your browser at `http://localhost:8501` 🎉

---

## 📁 Dataset

The dataset used is a financial transactions CSV with the following columns:

| Column | Description |
|---|---|
| `type` | Transaction type (PAYMENT, TRANSFER, CASH_OUT, etc.) |
| `amount` | Transaction amount |
| `oldbalanceOrg` | Sender balance before transaction |
| `newbalanceOrig` | Sender balance after transaction |
| `oldbalanceDest` | Receiver balance before transaction |
| `newbalanceDest` | Receiver balance after transaction |
| `isFraud` | Target: 1 = Fraud, 0 = Legitimate |

> 📌 The dataset is not included due to its large size (534+ MB). You can download a similar dataset from [Kaggle — Paysim Fraud Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)

---

## ⚠️ Common Issues & Fixes

**Issue: `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`**
```
This means your scikit-learn version doesn't match the one used to train the model.
Fix: Retrain the model in the same environment where you run the app.
```

**Issue: Streamlit not recognized in terminal**
```bash
pip install streamlit
# Make sure your conda environment is activated first!
```

**Issue: Model file not found**
```
Run main.ipynb completely first — it generates fraud_detection_pipeline.pkl
```

---

## 🔮 Future Improvements

- [ ] Try Random Forest and XGBoost for better precision
- [ ] Apply SMOTE oversampling to handle class imbalance differently
- [ ] Deploy to Streamlit Cloud (make it live on the internet!)
- [ ] Add transaction history and batch prediction feature
- [ ] Build a dashboard with real-time fraud analytics

---

## 👨‍💻 About Me

**Mrinmoy** — Data Science & AI/ML Student

I build real projects to learn real skills. This is my first end-to-end ML project and definitely not my last!

LinkedIn:-www.linkedin.com/in/mrinmoy-talukdar-5867ab3b9

Medium:-https://medium.com/@mrinmoytalukdargcu

---

<p align="center">Made with ❤️ and way too many debugging sessions at 2AM 🌙</p>