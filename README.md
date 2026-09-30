# Data-Driven Smartphone Prediction from Device Specifications Using Machine Learning
A complete end-to-end machine learning project that predicts smartphone prices based on technical specifications such as RAM, processor, camera, battery, display and connectivity features. Seven regression models were built, tuned and compared. The best model — **Gradient Boosting Regressor** — was deployed as an interactive web application using Streamlit.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [ML Pipeline](#ml-pipeline)
- [Feature Engineering](#feature-engineering)
- [Models Used](#models-used)
- [Model Performance](#model-performance)
- [Best Model Results](#best-model-results)
- [Screenshots](#screenshots)
- [Future Expansion](#future-expansion)
- [Author](#author)

---

## 📖 Project Overview

Smartphone prices vary widely across brands and specifications. Consumers and retailers often lack a reliable data-driven tool to estimate fair market prices based on technical specifications alone.

This project builds a supervised machine learning system that:
- Analyses and explores a dataset of 980 smartphones
- Engineers new features from existing specifications
- Trains and compares 7 regression models
- Selects the best model through cross-validated evaluation
- Deploys the model as a real-time web application using Streamlit

**Target Variable:** `price` (smartphone price in ₹)

---

## 📊 Dataset

| Property | Details |
|---|---|
| Total Records | 980 smartphones |
| Total Features | 22 columns |
| Numerical Features | 18 |
| Categorical Features | 4 (brand_name, model, processor_brand, os) |
| Price Range | ₹4,000 to ₹1,99,990 (after outlier removal) |
| Source | Smartphone specifications dataset |

**Key Features Used:**
- `brand_name`, `processor_brand`, `os`
- `ram_capacity`, `internal_memory`, `processor_speed`
- `battery_capacity`, `fast_charging`, `5G_or_not`
- `primary_camera_rear`, `primary_camera_front`
- `screen_size`, `refresh_rate`, `resolution_height`, `resolution_width`
- `avg_rating`, `num_cores`, `num_rear_cameras`
- `extended_memory_available`

---

## 📁 Project Structure

```
smartphone-price-prediction/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── smartphone_price_model.pkl      # Saved model bundle
├── ml_smartphone_price.ipynb       # Complete ML notebook (Google Colab)
├── test_model.py                   # Model testing script
├── .gitignore
└── README.md
```

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| ML Library | scikit-learn 1.6.1 |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Statistical Tests | Statsmodels, SciPy |
| Web App | Streamlit |
| Model Saving | Pickle |
| Development | Google Colab, VS Code |
| Version Control | Git, GitHub |

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/shravya184/smartphone-price-prediction.git
cd smartphone-price-prediction
```

**2. Create a virtual environment**
```bash
python -m venv .venv
```

**3. Activate the virtual environment**

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

Mac / Linux:
```bash
source .venv/bin/activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

**1.** Download `smartphone_price_model.pkl` from Google Drive and place it in the project folder alongside `app.py`

**2.** Run the Streamlit app:
```bash
python -m streamlit run app.py
```

**3.** Open your browser at: `http://localhost:8501`

**4.** Fill in the smartphone specifications and click **Predict Price** to get the estimated price

---

## 🔄 ML Pipeline

The complete machine learning pipeline follows these steps in order:

```
1. Load Dataset
        ↓
2. Exploratory Data Analysis (EDA)
        ↓
3. Missing Value Treatment
        ↓
4. Outlier Detection & Removal (Vertu)
        ↓
5. Linear Regression Assumption Checks
        ↓
6. Train-Test Split (80/20 Stratified)
        ↓
7. Feature Engineering
        ↓
8. Preprocessing (Impute → Encode → Scale)
        ↓
9. Train 7 Models (Base → CV → Tune → Tuned CV)
        ↓
10. Compare Models
        ↓
11. Select Best Model
        ↓
12. Full Validation
        ↓
13. Save & Deploy
```

### Preprocessing Steps
- **Missing Values:** Median imputation for numerical, Mode imputation for categorical
- **Outlier Removal:** Vertu phone (₹2,39,999+) removed — price driven by luxury branding, not specs
- **Log Transform:** `log1p(price)` applied as target to handle skewness. Predictions converted back via `expm1()`
- **Encoding:** One-Hot Encoding for categorical columns (`drop='first'` to avoid dummy variable trap)
- **Scaling:** StandardScaler for numerical columns
- **Data Leakage Prevention:** All preprocessing fit only on training data, applied to test data

---

## 🔧 Feature Engineering

Four new features were created from existing columns to capture price-driving interactions:

| New Feature | Formula | Why It Helps |
|---|---|---|
| `performance_score` | RAM × Processor Speed | Captures combined processing power — a key price driver |
| `camera_score` | Rear MP + Front MP | Combined camera quality indicator |
| `display_score` | (Resolution H × Resolution W) × Refresh Rate | Overall display quality score |
| `storage_to_ram` | Internal Memory ÷ RAM | Differentiates budget vs premium phone balance |

---

## 🤖 Models Used

Seven regression models were trained and compared:

| Model | Type | Role |
|---|---|---|
| Linear Regression | Linear | Baseline |
| Ridge Regression | Linear (L2) | Regularised baseline |
| Lasso Regression | Linear (L1) | Feature selection baseline |
| Decision Tree | Tree-based | Non-linear |
| Random Forest | Ensemble (Bagging) | Variance reduction |
| **Gradient Boosting** | **Ensemble (Boosting)** | **Best model ✅** |
| XGBoost | Ensemble (Boosting) | Optimised boosting |

### Evaluation Order for Each Model
Every model followed this standard evaluation order:
```
Base Train → Base Evaluate → Base CV → Hyperparameter Tuning → Tuned Evaluate → Tuned CV → Residual Plot → Conclusion
```

### Hyperparameter Tuning
- **Method:** GridSearchCV with 5-Fold Cross-Validation
- **Scoring:** R²
- **Gradient Boosting Parameters Tuned:** `n_estimators`, `max_depth`, `learning_rate`, `min_samples_leaf`

---

## 📈 Model Performance

| Model | Train R² | Test R² | Test RMSE | Test MAE | Mean CV R² |
|---|---|---|---|---|---|
| Linear Regression | 0.8808 | 0.7369 | 15,939 | 7,363 | 0.8430 |
| Ridge Regression | 0.8697 | 0.7463 | 15,650 | 7,383 | 0.8721 |
| Lasso Regression | 0.8541 | 0.7273 | 16,227 | 7,589 | 0.8635 |
| Decision Tree | 0.9524 | 0.8361 | 12,581 | 7,158 | 0.8521 |
| Random Forest | 0.9736 | 0.8032 | 13,786 | 6,112 | 0.9076 |
| **Gradient Boosting** | **—** | **0.8777** | **10,865** | **5,217** | **0.9169** |
| XGBoost | — | — | — | — | — |

> **Best model selected by:** Highest Mean CV R² combined with highest Test R²

---

## 🏆 Best Model Results

**Model:** Gradient Boosting Regressor

| Metric | Value |
|---|---|
| Test R² | 0.8777 |
| Test RMSE | ₹10,865 |
| Test MAE | ₹5,217 |
| Mean CV R² (Tuned) | 0.9169 |
| CV Std | 0.0128 |
| MAPE (mean) | 14.41% |
| APE (median) | 10.82% |

**20-Sample Validation:**

| Metric | Value |
|---|---|
| Validation R² | 0.9128 |
| Validation RMSE | ₹5,774 |
| Validation MAE | ₹3,941 |
| Validation MAPE | 14.39% |

**Key Findings:**
- Internal memory, RAM, processor speed and performance_score are the strongest price predictors
- Gradient Boosting's sequential error correction outperforms all other models on this dataset
- Log-transformation of price significantly improved accuracy across all models
- Median APE (~10.82%) gives a more honest view of typical prediction accuracy than Mean MAPE

---

## 🚀 Future Expansion

1. **Price Segment Modelling** — Train separate models for Budget (< ₹15K), Mid-range (₹15K–₹40K) and Premium (> ₹40K) to further reduce MAE
2. **Expand Dataset** — Add launch year, number of reviews, seller platform and storage variants
3. **Deep Learning** — Implement Neural Network (MLP) to capture deeper non-linear feature interactions
4. **Real-Time Price Scraping** — Integrate web scraping from Flipkart/Amazon to auto-update dataset monthly
5. **Cloud Deployment** — Deploy to Streamlit Cloud, Heroku or AWS EC2 for public access
6. **SHAP Explainability** — Add SHAP values to explain individual price predictions

---

## 👩‍💻 Author

**Raji (shravya184)**
- GitHub: [@shravya184](https://github.com/shravya184)
- Project: [smartphone-price-prediction](https://github.com/shravya184/smartphone-price-prediction)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

> **Note:** The model predicts approximate prices, not exact values. A typical prediction error (MAE) of ₹5,217 on prices ranging ₹4,000–₹1,99,990 represents a solid, practically useful result for a specification-based price estimator.
