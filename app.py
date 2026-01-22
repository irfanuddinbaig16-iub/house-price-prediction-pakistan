import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Prediction", layout="centered")

model = joblib.load("house_model.pkl")

st.title("House Price Prediction in Pakistan")

marla = st.text_input("Area (Marla)", "5")
bedrooms = st.text_input("Bedrooms", "3")
bathrooms = st.text_input("Bathrooms", "2")

# Static mapping (deploy safe)
CITY_LOCATIONS = {
    "Karachi": ["DHA", "Gulshan-e-Iqbal", "North Nazimabad", "Clifton"],
    "Lahore": ["DHA", "Johar Town", "Bahria Town", "Model Town"],
    "Islamabad": ["DHA", "G-10", "F-10", "Bahria Town"],
}

city = st.selectbox("City", list(CITY_LOCATIONS.keys()))
location = st.selectbox("Location", CITY_LOCATIONS[city])

if st.button("Predict Price"):
    try:
        marla_val = float(marla)
        bed_val = int(bedrooms)
        bath_val = int(bathrooms)

        if marla_val <= 0 or bed_val <= 0 or bath_val <= 0:
            st.error("Values must be greater than zero.")
            st.stop()

    except ValueError:
        st.error("Please enter valid numeric values.")
        st.stop()

    # Convert Marla → sqft (model expects Area_sqft)
    area_sqft = marla_val * 272.25

    input_df = pd.DataFrame([{
        "Area_sqft": area_sqft,
        "Bedrooms": bed_val,
        "Bathrooms": bath_val,
        "City": city,
        "Location": location,
    }])

    price = model.predict(input_df)[0]
    st.success(f"Estimated House Price: PKR {price:,.0f}")
