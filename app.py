import streamlit as st
import numpy as np
import pickle

# Load model + scaler
model, scaler = pickle.load(open("breast_cancer.pkl", "rb"))

st.title("🔬 Breast Cancer Prediction App")
st.write("Enter the tumor details below:")

# Example inputs (you can expand to more features)
mean_radius = st.number_input("Mean Radius")
mean_texture = st.number_input("Mean Texture")
mean_perimeter = st.number_input("Mean Perimeter")
mean_area = st.number_input("Mean Area")

if st.button("Predict"):
    # Prepare input
    features = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area]])
    features_scaled = scaler.transform(
        np.pad(features, ((0,0),(0,26)), 'constant')  # pad to match original 30 features
    )
    
    # Prediction
    prediction = model.predict(features_scaled)[0]
    result = "Benign (No Cancer)" if prediction == 1 else "Malignant (Cancer)"
    st.success(f"Prediction: {result}")
