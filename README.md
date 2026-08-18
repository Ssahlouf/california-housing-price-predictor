# 🏡 California House Price Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit)
![R2 Score](https://img.shields.io/badge/R²-0.82-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end machine learning project that predicts California median house prices using the California Housing dataset. Includes full EDA, feature engineering, model comparison, and an interactive Streamlit web app.

---

## 📌 Project Overview

This project walks through the complete data science workflow:

- **Exploratory Data Analysis (EDA)** — understanding distributions, correlations, and data quality
- **Data Cleaning** — handling missing values, removing capped values and outliers
- **Feature Engineering** — creating meaningful ratios from raw features
- **Model Training & Comparison** — Linear Regression, Random Forest, and XGBoost
- **Model Evaluation** — MAE, RMSE, and R² metrics
- **Deployment** — interactive Streamlit app for real-time predictions

---

## 📊 Model Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $43,257 | $58,503 | 0.6524 |
| Random Forest | $30,045 | $44,820 | 0.7960 |
| **XGBoost** ✅ | **$28,495** | **$42,061** | **0.8203** |

> XGBoost was selected as the final model with an R² of **0.82** and a mean absolute error of **$28,495**.

---

## 🗂️ Project Structure

```
california-housing-price-predictor/
│
├── Data/
│   └── housing.csv                  # Raw dataset
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb    # Full EDA and modeling walkthrough
│
├── models/
│   ├── best_model.pkl               # Trained XGBoost model
│   └── scaler.pkl                   # Fitted StandardScaler
│
├── app.py                           # Streamlit web app
├── requirements.txt                 # Project dependencies
└── README.md
```

---

## 🔍 Key Findings

- **`median_income`** is the strongest predictor of house value (correlation: 0.65)
- **`total_rooms`, `total_bedrooms`, `households`, `population`** are highly correlated (0.86–0.97) — indicating multicollinearity
- The dataset had prices **capped at $500,000** — these were removed to avoid misleading the model
- **207 missing values** in `total_bedrooms` — filled using the median (robust to outliers)

---

## ⚙️ Feature Engineering

Three new features were created from existing columns:

| Feature | Formula | Why |
|---|---|---|
| `rooms_per_household` | `total_rooms / households` | Quality of housing |
| `bedrooms_per_room` | `total_bedrooms / total_rooms` | Space ratio |
| `population_per_household` | `population / households` | Crowding level |

Outliers in engineered features were removed using domain-based thresholds.

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/Ssahlouf/california-housing-price-predictor.git
cd california-housing-price-predictor
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

### 5. Or explore the notebook
```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas & numpy** — data manipulation
- **matplotlib & seaborn** — visualizations
- **scikit-learn** — preprocessing, Linear Regression, Random Forest
- **XGBoost** — best performing model
- **Streamlit** — web app deployment
- **joblib** — model serialization

---

## 🌐 Live Demo
👉 [Try the app here](https://california-housing-price-predictor-mojvzmskvxpjkg3agt6dh7.streamlit.app/)

## 👤 Author

**Ssahlouf**  
[GitHub](https://github.com/Ssahlouf) 

---

## 📄 License

This project is licensed under the MIT License.
