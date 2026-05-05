# Heart Disease Prediction App
A clinical decision support tool built with **Streamlit** and **Scikit-Learn**. This application utilizes the **UCI Cleveland Heart Disease dataset** to predict the probability of heart disease in patients based on 13 clinical features.

[Link to streamlit app](https://heartdiseaseuci-sjzzhaufc3j9uaesvtd6ft.streamlit.app/ "Streamlit app")

## Project Overview
This project implements a full machine learning pipeline, from data preprocessing and feature selection to model deployment. The core objective is to maximize **Recall**—minimizing false negatives to ensure potential heart disease cases are not overlooked in a clinical setting.

**Key Features:**
* **Automated Preprocessing:** Handles missing values via `KNNImputer` and scales features via `StandardScaler`.

* **Feature Selection:** Utilizes `SelectKBest` with `F-value` scoring to identify the most significant diagnostic indicators.

* **Explainable:** The app extracts model coefficients to show the **Top 3 Diagnostic Drivers** (e.g., how age or cholesterol contributed to the specific risk score).

* **Robust Evaluation:** Models were trained using `RepeatedStratifiedKFold` cross-validation to ensure stability across different data splits.

## Tech Stack

* **Frontend:** Streamlit

* **Machine Learning:** Scikit-Learn

## Model Performance


* **Data Manipulation:** Pandas, NumPy

* **Model Tracking:** Joblib

* **Modeling Experiments:** Logistic Regression (Champion), Random Forest, XGBoost, and SVM.

## Model Performance
After a tournament-style grid search, **Logistic Regression** was selected as the champion model due to its high recall and interpretability.

Metric | Score
--- | --- 
Primary Metric (Recall) | ~85-90% (Variable by fold)
Validation Strategy | 5-Fold CV, Repeated 3x
Optimization Goal | Recall (Sensitivity)

## Local Installation

Clone the repository:
```Bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```
Install dependencies:
```Bash
   pip install -r requirements.txt
```
Run the application:
```Bash
streamlit run app.py
```

## Data Dictionary
The model expects the following clinical inputs:

* **Age:** Patient's age in years.

* **Sex:** Male (1) or Female (0).

* **CP:** Chest pain type (Typical, Atypical, Non-anginal, Asymptomatic).

* **Trestbps:** Resting blood pressure (mm Hg).

* **Chol:** Serum cholesterol (mg/dl).

* **Fbs:** Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).

* **Restecg:** Resting electrocardiographic results.

* **Thalach:** Maximum heart rate achieved.

* **Exang:** Exercise-induced angina (1 = yes; 0 = no).

* **Oldpeak:** ST depression induced by exercise relative to rest.

* **Slope:** The slope of the peak exercise ST segment.

* **Ca:** Number of major vessels (0-3) colored by fluoroscopy.

* **Thal:** Thalassemia (Normal, Fixed defect, Reversible defect).










