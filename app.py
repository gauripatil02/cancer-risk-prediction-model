import streamlit as st
import pickle
import numpy as np
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Page configuration
st.set_page_config(page_title="Cancer Risk Prediction", layout="centered")

# Load assets
@st.cache_resource
def load_assets():
    with open('cancer_model.pkl', 'rb') as f: m = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: s = pickle.load(f)
    return m, s

model, scaler = load_assets()

# UI State Management
if 'view' not in st.session_state: st.session_state.view = 'form'

st.markdown("### Cancer Risk Prediction")

if st.session_state.view == 'form':
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 18, 100, 45)
        gen = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x else "Female")
        bmi = st.number_input("BMI", 10.0, 50.0, 24.0)
        smk = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with c2:
        rig = st.selectbox("Genetic Risk", [0, 1, 2], format_func=lambda x: ["Low", "Mid", "High"][x])
        his = st.selectbox("Family History", [0, 1], format_func=lambda x: "Yes" if x else "No")
        act = st.number_input("Activity (Hrs/Wk)", 0, 10, 5)
        alc = st.number_input("Alcohol (Units/Wk)", 0, 5, 1)

    if st.button("RESULT"):
        st.session_state.data = np.array([[age, gen, bmi, smk, rig, act, alc, his]])
        st.session_state.view = 'result'
        st.rerun()

else:
    # Result View
    scaled = scaler.transform(st.session_state.data)
    pred = model.predict(scaled)[0]
    conf = model.predict_proba(scaled)[0]

    st.markdown("---")
    st.metric("Assessment Result", "Positive" if pred else "Negative")
    st.write(f"Confidence: {conf[pred]*100:.1f}%")
    
    if st.button("Return to Form"):
        st.session_state.view = 'form'
        st.rerun()