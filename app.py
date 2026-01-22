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
ISLAMABAD_LOCATIONS = [
    "DHA Phase 1", "DHA Phase 2",
    "Bahria Town Phase 1", "Bahria Town Phase 2",
    "F-6", "F-7", "F-8", "F-10", "F-11",
    "G-6", "G-7", "G-8", "G-9", "G-10", "G-11", "G-13",
    "H-8", "H-9", "H-10", "H-11",
    "I-8", "I-9", "I-10",
]

LAHORE_LOCATIONS = [
    "DHA Phase 1", "DHA Phase 2", "DHA Phase 3",
    "DHA Phase 4", "DHA Phase 5", "DHA Phase 6",
    "Bahria Town",
    "Johar Town",
    "Model Town",
    "Garden Town",
    "Wapda Town",
    "Faisal Town",
    "Iqbal Town",
]

KARACHI_LOCATIONS = [
    "DHA Phase 1", "DHA Phase 2", "DHA Phase 4",
    "Clifton",
    "Gulshan-e-Iqbal",
    "North Nazimabad",
    "PECHS",
    "Gulistan-e-Johar",
    "Malir",
]
CITY_LOCATIONS = {
    "Islamabad": ISLAMABAD_LOCATIONS,
    "Lahore": LAHORE_LOCATIONS,
    "Karachi": KARACHI_LOCATIONS,
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
