# Heart Disease Prediction App
A clinical decision support tool built with **Streamlit** and **Scikit-Learn**. This application utilizes the **UCI Cleveland Heart Disease dataset** to predict the probability of heart disease in patients based on 13 clinical features.

[Link to streamlit app](https://heartdiseaseuci.streamlit.app/ "Streamlit app")

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

### Test data
Some data to see how the app works


**1. High Risk Profile**

Field | Value | Reason for Risk
--- | --- | --- 
Birth date | 1965-05-15 (Age ~61) | Higher age bracket
Sex | Male (1) | Statistically higher risk in dataset
Chest Pain | TypeAsymptomatic (3) | Often linked to silent ischemia
Resting BP | 150 | Hypertension
Cholestero| l280 |High cholesterol
Fasting Blood Sugar | 145 (Resulting in 1) | Hyperglycemia
Resting ECG | Left ventricular hypertrophy (2) | Cardiac stress indicator
Max Heart Rate | 110 | Low max heart rate for age
Ex. Induced Angina | Checked (True) | Strong predictor of disease
Oldpeak | 2.5 | Significant ST depression
Slope | Downsloping (3) | High-risk ST segment
Major Vessels (ca) | 2 | Blockage in multiple vessels
Thal | Reversible defect (7) | Poor blood flow during stress

**2. Low Risk Profile**

Field | Value | Reason for Risk
--- | --- | ---
Birth date | 1995-10-20 (Age ~30)| Low age
Sex | Female (0) | Lower statistical risk in dataset
Chest Pain Type | Typical angina (0) | Usually less predictive than Type 3
Resting BP | 115 | Optimal BP
Cholesterol | 190 | Healthy range
Fasting Blood Sugar | 90 (Resulting in 0) | Normal glucose
Resting ECG | Normal (0) | Healthy baseline
Max Heart Rate | 185 | High cardiovascular capacity
Ex. Induced Angina | Unchecked (False) | No pain during exertion
Oldpeak | 0.0 | No ST depression
Slope | Upsloping (1) | Healthy recovery segment
Major Vessels (ca) | 0 | No visible vessel blockage
Thal | Normal (3) | Normal blood flow



