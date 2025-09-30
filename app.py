import streamlit as st
import pandas as pd
import joblib

st.title("Women Safety Prediction App")

# Load trained model
model = joblib.load("women_safety_model.pkl")
train_columns = joblib.load("train_columns.pkl")  # optional if you one-hot encoded

# Sidebar for user input
st.sidebar.header("Enter Area Details")
Area = st.sidebar.selectbox("Area (1-Urban,2-Suburban,3-Rural)", [1,2,3])
Zone = st.sidebar.slider("Zone (1-26)", 1, 26)
People_Frequency = st.sidebar.slider("People Frequency (1-Low,2-Medium,3-High)", 1, 3)
Is_Police_Station = st.sidebar.selectbox("Police Station Nearby?", [1,2])
Is_Bar = st.sidebar.selectbox("Bar Nearby?", [1,2])
Tier = st.sidebar.selectbox("Tier (1-Low,2-Medium,3-High)", [1,2,3])
Residence_Level = st.sidebar.selectbox("Residence Level (1-Low,2-Medium,3-High)", [1,2,3])

# Function to preprocess input
def preprocess_input(Area, Zone, People_Frequency, Is_Police_Station, Is_Bar, Tier, Residence_Level):
    df = pd.DataFrame({
        'Area':[Area],
        'Zone':[Zone],
        'People.Frequency':[People_Frequency],
        'Is.Police_Station':[Is_Police_Station],
        'Is.Bar':[Is_Bar],
        'Tier':[Tier],
        'Residence.Level':[Residence_Level]
    })
    # Align columns if one-hot encoding was used
    df = pd.get_dummies(df)
    df = df.reindex(columns=train_columns, fill_value=0)
    return df

# Predict button
if st.sidebar.button("Predict"):
    input_df = preprocess_input(Area, Zone, People_Frequency, Is_Police_Station, Is_Bar, Tier, Residence_Level)
    prediction = model.predict(input_df)
    if prediction[0]==1:
        st.success("This area is predicted SAFE ")
    else:
        st.error("This area is predicted UNSAFE ")
