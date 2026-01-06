import streamlit as st
import pickle
import numpy as np

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="ML Model Prediction",
    page_icon="🤖",
    layout="centered"
)

st.title("🔮 ML Model Prediction App")

# --------------------------------
# Load Model
# --------------------------------
@st.cache_resource
def load_model():
    with open("Knn1.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.success("✅ Model Loaded Successfully")

# --------------------------------
# User Input
# (example: 2 features)
# --------------------------------
Age = st.number_input("Enter Age", value=0.0)
Estimated_salary = st.number_input("Enter Estimated Salary", value=0.0)

# --------------------------------
# Prediction
# --------------------------------
if st.button("Predict"):
    input_data = np.array([[Age,Estimated_salary]])
    prediction = model.predict(input_data)

    st.subheader("📊 Prediction Result")
    st.write(prediction[0])