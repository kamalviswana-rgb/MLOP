import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------------
# Load Model
# -------------------------------------

MODEL_PATH = Path(__file__).parent / "tourism_model.pkl"

model = joblib.load(MODEL_PATH)

st.title("Tourism Package Prediction")

st.write(
    "Predict whether a customer is likely to purchase a tourism package."
)

# -------------------------------------
# User Inputs
# -------------------------------------

age = st.number_input("Age", min_value=18, max_value=100, value=35)

typeofcontact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

citytier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Small Business",
        "Large Business",
        "Free Lancer"
    ]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

numberofpersonvisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

preferredpropertystar = st.selectbox(
    "Preferred Property Star",
    [1, 2, 3, 4, 5]
)

maritalstatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

numberoftrips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=50,
    value=2
)

passport = st.selectbox(
    "Passport",
    [0, 1]
)

pitchsatisfactionscore = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

owncar = st.selectbox(
    "Own Car",
    [0, 1]
)

numberofchildrenvisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

monthlyincome = st.number_input(
    "Monthly Income",
    min_value=1000,
    value=30000
)

# -------------------------------------
# Prediction
# -------------------------------------

if st.button("Predict"):

    input_df = pd.DataFrame({
        "Age": [age],
        "TypeofContact": [typeofcontact],
        "CityTier": [citytier],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [numberofpersonvisiting],
        "PreferredPropertyStar": [preferredpropertystar],
        "MaritalStatus": [maritalstatus],
        "NumberOfTrips": [numberoftrips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitchsatisfactionscore],
        "OwnCar": [owncar],
        "NumberOfChildrenVisiting": [numberofchildrenvisiting],
        "Designation": [designation],
        "MonthlyIncome": [monthlyincome]
    })

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success(
            "Customer is likely to purchase the tourism package."
        )
    else:
        st.warning(
            "Customer is unlikely to purchase the tourism package."
        )
