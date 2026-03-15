import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Load pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load('models/readmission_pipeline.pkl')

pipeline = load_pipeline()

# Title
st.title("🏥 Hospital Readmission Risk Predictor")
st.markdown("### Predicting 30-day readmission risk for diabetic patients")
st.divider()

# ── Sidebar - Patient Input ──
st.sidebar.header("👤 Patient Information")

age = st.sidebar.slider("Age", 10, 100, 65)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
time_in_hospital = st.sidebar.slider("Days in Hospital", 1, 14, 4)
num_lab_procedures = st.sidebar.slider("Number of Lab Procedures", 1, 132, 43)
num_procedures = st.sidebar.slider("Number of Procedures", 0, 6, 1)
num_medications = st.sidebar.slider("Number of Medications", 1, 81, 15)
number_inpatient = st.sidebar.slider("Prior Inpatient Visits", 0, 21, 0)
number_emergency = st.sidebar.slider("Prior Emergency Visits", 0, 76, 0)
number_outpatient = st.sidebar.slider("Prior Outpatient Visits", 0, 42, 0)
number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 16, 7)
insulin = st.sidebar.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
change = st.sidebar.selectbox("Medication Change During Visit", ["No", "Yes"])
diabetesMed = st.sidebar.selectbox("On Diabetes Medication", ["Yes", "No"])

# ── Out-of-Distribution Detection ──
ood_warnings = []

if age < 40:
    ood_warnings.append(f"⚠️ Patient age ({age}) is below the typical training range (40-90). Prediction reliability may be reduced.")
if num_medications < 3:
    ood_warnings.append(f"⚠️ Very few medications ({num_medications}) — unusual for diabetic patients in training data.")
if diabetesMed == "No" and insulin == "No":
    ood_warnings.append("⚠️ Patient is on neither diabetes medication nor insulin — rare profile in training data.")

# ── Confidence Score ──
def get_confidence(prob, age, diabetesMed, insulin, ood_warnings):
    confidence = "High"
    confidence_color = "#2ecc71"
    confidence_note = "Model is reliable for this patient profile."

    if len(ood_warnings) >= 2:
        confidence = "Low"
        confidence_color = "#e74c3c"
        confidence_note = "⚠️ Multiple out-of-distribution factors detected. Human review recommended."
    elif len(ood_warnings) == 1:
        confidence = "Moderate"
        confidence_color = "#f39c12"
        confidence_note = "🔍 Some unusual patient characteristics. Use prediction with caution."
    elif 0.18 <= prob <= 0.22:
        confidence = "Moderate"
        confidence_color = "#f39c12"
        confidence_note = "🔍 Prediction is near decision threshold. Consider clinical judgment."

    return confidence, confidence_color, confidence_note

# ── Feature Engineering ──
gender_enc = 1 if gender == "Male" else 0
insulin_enc = ["No", "Steady", "Up", "Down"].index(insulin)
change_enc = 1 if change == "Yes" else 0
diabetesMed_enc = 1 if diabetesMed == "Yes" else 0

hospital_utilization_score = number_inpatient + number_emergency + number_outpatient
num_med_changed = 1 if change == "Yes" else 0
num_med_prescribed = num_medications
treatment_intensity = time_in_hospital + num_lab_procedures + num_procedures + num_medications
patient_complexity = number_diagnoses * time_in_hospital
primary_diag_diabetes = 1

if age <= 30:
    age_risk_group = 0
elif age <= 60:
    age_risk_group = 1
elif age <= 75:
    age_risk_group = 2
else:
    age_risk_group = 3

# ── Build Input DataFrame ──
input_data = pd.DataFrame({
    'race': [2], 'gender': [gender_enc], 'age': [age],
    'admission_type': [1], 'discharge_disposition': [1],
    'admission_source': [7], 'time_in_hospital': [time_in_hospital],
    'medical_specialty': [8], 'num_lab_procedures': [num_lab_procedures],
    'num_procedures': [num_procedures], 'num_medications': [num_medications],
    'number_outpatient': [number_outpatient], 'number_emergency': [number_emergency],
    'number_inpatient': [number_inpatient], 'diag_1': [250],
    'number_diagnoses': [number_diagnoses], 'metformin': [1],
    'repaglinide': [1], 'nateglinide': [1], 'glimepiride': [1],
    'glipizide': [1], 'glyburide': [1], 'pioglitazone': [1],
    'rosiglitazone': [1], 'acarbose': [0], 'miglitol': [1],
    'insulin': [insulin_enc], 'change': [change_enc],
    'diabetesMed': [diabetesMed_enc],
    'hospital_utilization_score': [hospital_utilization_score],
    'num_med_changed': [num_med_changed],
    'num_med_prescribed': [num_med_prescribed],
    'treatment_intensity': [treatment_intensity],
    'patient_complexity': [patient_complexity],
    'primary_diag_diabetes': [primary_diag_diabetes],
    'age_risk_group': [age_risk_group]
})

# ── Prediction ──
prob = pipeline.predict_proba(input_data)[0][1]
risk = "HIGH RISK 🔴" if prob >= 0.2 else "LOW RISK 🟢"
confidence, confidence_color, confidence_note = get_confidence(
    prob, age, diabetesMed, insulin, ood_warnings)

# ── OOD Warnings Banner ──
if ood_warnings:
    st.warning("**Model Reliability Warnings Detected:**")
    for w in ood_warnings:
        st.write(w)
    st.divider()

# ── Main Metrics ──
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Readmission Probability", f"{prob*100:.1f}%")
with col2:
    st.metric("Risk Level", risk)
with col3:
    st.metric("Decision Threshold", "20%")
with col4:
    st.metric("Prediction Confidence", confidence)

st.divider()

# ── Human Review Flag ──
if confidence == "Low":
    st.error("""
    🚨 **HUMAN REVIEW REQUIRED**
    This patient profile is significantly outside the model's 
    training distribution. Please have a clinician manually 
    assess readmission risk before making any decisions.
    """)
elif confidence == "Moderate":
    st.warning("""
    🔍 **CLINICAL JUDGMENT ADVISED**
    This prediction has moderate confidence. 
    Please consider additional clinical factors.
    """)

# ── Risk Gauge ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Risk Assessment")
    risk_color = "#e74c3c" if prob >= 0.2 else "#2ecc71"
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(['Readmission Risk'], [prob*100], color=risk_color)
    ax.barh(['Readmission Risk'], [100 - prob*100],
            left=[prob*100], color='#ecf0f1')
    ax.axvline(x=20, color='orange', linestyle='--',
               linewidth=2, label='Threshold (20%)')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Risk (%)')
    ax.legend()
    ax.set_title('Patient Readmission Risk')
    st.pyplot(fig)

with col2:
    st.subheader("📋 Clinical Recommendation")
    if confidence == "Low":
        st.error("""
        🚨 **HUMAN REVIEW REQUIRED**
        - Do NOT rely solely on model prediction
        - Assign to senior clinician for assessment
        - Document reasoning for final decision
        """)
    elif prob >= 0.2:
        st.error("""
        ⚠️ **HIGH RISK PATIENT**

        Recommended Actions:
        - Schedule follow-up within 7 days
        - Review medication plan
        - Arrange discharge support
        - Consider social worker referral
        """)
    else:
        st.success("""
        ✅ **LOW RISK PATIENT**

        Recommended Actions:
        - Standard discharge protocol
        - Routine follow-up in 30 days
        - Provide self-care instructions
        """)

st.divider()

# ── Confidence Explanation ──
st.subheader("🎯 Prediction Confidence Details")
st.info(f"""
**Confidence Level: {confidence}**

{confidence_note}

*Model trained on diabetic patients aged 40-90. 
Best performance for elderly patients with active diabetes management.*
""")

st.divider()

# ── Patient Summary ──
st.subheader("👤 Patient Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Age", age)
col2.metric("Days in Hospital", time_in_hospital)
col3.metric("Prior Inpatient Visits", number_inpatient)
col4.metric("Number of Diagnoses", number_diagnoses)

st.divider()
st.caption("Built with ❤️ using XGBoost + Streamlit | Healthcare Readmission Pipeline Project")