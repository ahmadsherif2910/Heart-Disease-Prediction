import streamlit as st
import datetime
import pandas as pd
import joblib

today = datetime.date.today()


@st.cache_resource
def load_model():
    model_path = "LogisticRegression.pkl"
    return joblib.load(model_path)

def get_clinical_explanation(pipeline, patient_data):
    """
    Refactored version of your explanation logic
    returns data instead of printing.
    """
    # 1. Get Probability
    prob = pipeline.predict_proba(patient_data)[0][1]

    # 2. Extract components
    classifier = pipeline.named_steps['classifier']
    selector = pipeline.named_steps['selector']
    preprocessor = pipeline.named_steps['preprocessor']

    # 3. Transform data
    transformed_data = preprocessor.transform(patient_data)
    selected_data = selector.transform(transformed_data)

    # 4. Calculate Contribution
    contributions = classifier.coef_[0] * selected_data[0]
    all_names = preprocessor.get_feature_names_out()
    selected_names = all_names[selector.get_support()]

    # 5. Map and Sort
    importance = pd.Series(contributions, index=selected_names)
    top_drivers = importance.abs().sort_values(ascending=False).head(3)

    drivers_list = []
    for name, val in top_drivers.items():
        clean_name = name.split('__')[-1]
        direction = "🔴 Increasing risk" if importance[name] > 0 else "🟢 Decreasing risk"
        drivers_list.append(f"**{clean_name}**: {direction}")

    # Risk Level logic
    risk_level = "High" if prob > 0.8 else ("Moderate" if prob > 0.5 else "Low")

    return {
        "prob": prob,
        "risk_level": risk_level,
        "drivers": drivers_list,
        "features_count": selector.k
    }


model = load_model()

st.title("Heart Disease Prediction App")
st.write("Enter patient information below:")

cp_values = {
    0:"Typical angina",
    1:"Atypical angina",
    2:"Non-angina",
    3:"Asymptomatic",
}

restecg_values = {
    0:"Normal",
    1:"ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)",
    2:"Probable or definite left ventricular hypertrophy by Estes' criteria",
}

slope_values = {
    1:"Upsloping",
    2:"Flat",
    3:"Downsloping",
}

thal_values = {
    3:"Normal",
    6:"Fixed defect",
    7:"Reversible defect",
}

age = st.date_input(label="Birth date",value=None,max_value=today.replace(year=today.year-18),min_value=today.replace(year=today.year-120))
sex = st.selectbox("Sex", [0, 1],index=None, format_func=lambda x: "Female" if x == 0 else "Male",placeholder="Choose your sex")
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3],index=None,format_func=lambda x:cp_values.get(x),placeholder="Choose Chest Pain Type (cp)")
trestbps = st.number_input("Resting Blood Pressure", 30, 260, None,placeholder="Enter resting blood pressure (mm Hg)")
chol = st.number_input("Cholesterol", 100, 500, None,placeholder="Enter cholesterol (mg/dl)")
fbs = st.number_input("Fasting Blood Sugar",min_value=20,max_value=600,value=None,placeholder="Enter fasting blood sugar (mg/dl)")
restecg = st.selectbox("Resting ECG", [0, 1, 2],index=None,format_func=lambda x:restecg_values.get(x),placeholder="Choose a Resting ECG type")
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, None,placeholder="Enter max heart rate achieved")
exang = st.checkbox("Exercise Induced Angina")
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, None,placeholder="Enter ST depression induced by exercise relative to rest (Oldpeak)")
slope = st.selectbox("Slope", [1, 2,3], index=None, format_func=lambda x:slope_values.get(x),placeholder="Choose the slope of the peak exercise ST segment")
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3],index = None,placeholder="Choose number of major vessels (0-3) colored by flourosopy")
thal = st.selectbox("Thal", [3,6,7],index=None,format_func=lambda x:thal_values.get(x),placeholder="Choose thal type")

if fbs is not None:
    fbs = 1 if fbs > 120 else 0

if age is not None:
    age = today.year - age.year - (
        (today.month, today.day) < (age.month, age.day)
    )

missing_fields = []

if age is None:
    missing_fields.append("Birth date")

if sex is None:
    missing_fields.append("Sex")

if cp is None:
    missing_fields.append("Chest Pain Type")

if trestbps is None:
    missing_fields.append("Resting Blood Pressure")

if chol is None:
    missing_fields.append("Cholesterol")

if fbs is None:
    missing_fields.append("Fasting Blood Sugar")

if restecg is None:
    missing_fields.append("Resting ECG")

if thalach is None:
    missing_fields.append("Max Heart Rate Achieved")

if oldpeak is None:
    missing_fields.append("Oldpeak")

if slope is None:
    missing_fields.append("Slope")

if ca is None:
    missing_fields.append("Number of Major Vessels")

if thal is None:
    missing_fields.append("Thal")

predict_disabled = len(missing_fields) > 0

if predict_disabled:
    st.warning(
        "Please complete the following fields:\n\n- " +
        "\n- ".join(missing_fields)
    )

if st.button("Predict", disabled=predict_disabled):
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])
    # Get explanation data
    results = get_clinical_explanation(model, input_data)

    st.divider()
    st.subheader("Clinical Summary")

    # Display Risk with Metrics
    cols = st.columns(2)
    cols[0].metric("Risk Assessment", results["risk_level"])
    cols[1].metric("Disease Probability", f"{results['prob']:.1%}")

    # Display Drivers
    st.write("### Top Diagnostic Drivers")
    for driver in results["drivers"]:
        st.write(driver)

    # Footer Info
    st.info(f"Model Integrity: Optimized via {results['features_count']} features.")

    # Validation Recall (if available)
    val_score = getattr(model, 'val_score_', None)
    if val_score:
        st.caption(f"Historical Validation Recall: {val_score:.2%}")

