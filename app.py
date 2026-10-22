"""
Heart Disease Prediction System - Streamlit Web App
----------------------------------------------------
Loads the Support Vector Classifier trained in the Heart Disease Prediction
System notebook (heart_disease_svc_trained_model.pkl) and recreates the exact
same LabelEncoders (fit on the same fixed category lists as Step 5.1 of the
notebook), so predictions match the notebook exactly.

Run locally:  streamlit run app.py
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model (same file the notebook produced in Step 6.4)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "heart_disease_svc_trained_model.pkl"

# Same category lists used to train the LabelEncoders in the notebook (Step 5.1).
# The order of the keys is the order of the model's input features.
CATEGORIES = {
    "ChestPain": ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
    "ExerciseAngina": ["Yes", "No"],
    "MajorVessels": ["Zero", "One", "Two", "Three"],
    "Thal": ["Normal", "Fixed Defect", "Reversible Defect"],
}


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    """
    LabelEncoder numbers categories in sorted (alphabetical) order, so fitting
    on the same fixed lists as the notebook reproduces exactly the same codes.
    """
    return {attribute: LabelEncoder().fit(values) for attribute, values in CATEGORIES.items()}


model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("🫀 Heart Disease Prediction System")
st.markdown(
    """
    Enter a patient's test results and the trained Support Vector Classifier (SVC)
    will predict whether heart disease is present.

    *Built from 297 patients in the UCI Cleveland Heart Disease dataset.
    Accuracy on 60 held-out test patients: **85%**.*
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    chest_pain = st.selectbox(
        "Chest pain type",
        CATEGORIES["ChestPain"],
        help="Typical angina: classic heart-related chest pain. Atypical angina: some but not "
             "all typical features. Non-anginal pain: chest pain not caused by the heart. "
             "Asymptomatic: no chest pain.",
    )
    exercise_angina = st.selectbox(
        "Exercise-induced angina",
        ["No", "Yes"],
        help="Did the patient get chest pain during the exercise stress test?",
    )
with col2:
    major_vessels = st.selectbox(
        "Major vessels colored by fluoroscopy",
        CATEGORIES["MajorVessels"],
        help="Number of major blood vessels (0-3) colored by fluoroscopy.",
    )
    thal = st.selectbox(
        "Thallium stress test result",
        CATEGORIES["Thal"],
        help="Normal: normal blood flow. Fixed defect: no blood flow to part of the heart, "
             "at rest and under stress. Reversible defect: blood flow drops only under stress.",
    )

if st.button("Predict heart disease", type="primary", width="stretch"):
    user_input = pd.DataFrame({
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
        "MajorVessels": [major_vessels],
        "Thal": [thal],
    })

    encoded = pd.DataFrame({
        attribute: encoders[attribute].transform(user_input[attribute]) for attribute in CATEGORIES
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### HEART DISEASE")
    else:
        st.success("### NO HEART DISEASE")

    with st.expander("See the label-encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Student project for Machine Learning Fundamentals (AIC354). The model uses only four "
    "attributes from a small public dataset, so its predictions are not a medical diagnosis."
)
