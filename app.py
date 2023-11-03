import streamlit as st
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer

with open(r"C:\Users\yogit\Desktop\MP\rf_model.pkl", "rb") as f:
    model = pickle.load(f)

normal_ranges = {
    "TT4": (4.5, 12.0),
    "T4U": (25, 35),
    "T3": (60, 200),
    "TSH": (0.55, 4.78)
}

st.title("Thyroid Disorder Detection")
st.write("Enter the input thyroid hormone levels:")

age = st.number_input("Age", min_value=0, max_value=100, value=25)
sex = st.selectbox("Sex", ["Male", "Female"])
pregnant = st.selectbox("Pregnant", ["Yes", "No"])
thyroid_surgery = st.selectbox("Thyroid Surgery", ["Yes", "No"])
tt4 = st.number_input("TT4 (4.5-12.0 μg/dL)", min_value=0.0, max_value=30.0, value=8.0, step=0.1)
t4u = st.number_input("T4U (25-35)", min_value=0.0, max_value=50.0, value=30.0, step=0.1)
t3 = st.number_input("T3 (60-200)", min_value=0.0, max_value=300.0, value=100.0, step=0.1)
tsh = st.number_input("TSH (0.55-4.78)", min_value=0.0, max_value=100.0, value=2.0, step=0.01)

normal_tt4 = normal_ranges["TT4"][0] <= tt4 <= normal_ranges["TT4"][1]
normal_t4u = normal_ranges["T4U"][0] <= t4u <= normal_ranges["T4U"][1]
normal_t3 = normal_ranges["T3"][0] <= t3 <= normal_ranges["T3"][1]
normal_tsh = normal_ranges["TSH"][0] <= tsh <= normal_ranges["TSH"][1]

user_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],  
    'pregnant': [pregnant],
    'thyroid_surgery': [thyroid_surgery],
    'TT4': [tt4],
    'T4U': [t4u],
    'T3': [t3],
    'TSH': [tsh],
})

user_data.sex.replace({'Female': 2, 'Male': 1}, inplace=True)

user_data["sex"].fillna(round(user_data["sex"].mean()), inplace=True)

user_data['on_thyroxine'] = None  
user_data['query_on_thyroxine'] = None
user_data['on_antithyroid_medication'] = None  
user_data['sick'] = None
user_data['I131_treatment'] = None  
user_data['query_hypothyroid'] = None
user_data['query_hyperthyroid'] = None  
user_data['lithium'] = None
user_data['goitre'] = None  
user_data['tumor'] = None
user_data['hypopituitary'] = None  
user_data['psych'] = None

imputer = SimpleImputer(strategy='mean')

user_data = pd.get_dummies(user_data, columns=['pregnant', 'thyroid_surgery'])
model_features = set(user_data.columns)
user_data.fillna(0, inplace=True)

if st.button("Predict Thyroid Status"):
        prediction = model.predict(user_data)[0]

        normal_tt4 = normal_ranges["TT4"][0] <= tt4 <= normal_ranges["TT4"][1]
        normal_t4u = normal_ranges["T4U"][0] <= t4u <= normal_ranges["T4U"][1]
        normal_t3 = normal_ranges["T3"][0] <= t3 <= normal_ranges["T3"][1]
        normal_tsh = normal_ranges["TSH"][0] <= tsh <= normal_ranges["TSH"][1]

        if normal_tt4 and normal_t4u and normal_t3 and normal_tsh:
            status = "Normal"
        elif tt4 > normal_ranges["TT4"][1] or t4u > normal_ranges["T4U"][1] or t3 > normal_ranges["T3"][1] or tsh > normal_ranges["TSH"][1]:
            status = "Hyperthyroidism"
        else:
            status = "Hypothyroidism"

        st.subheader("Thyroid Status:")
        st.write(f"The thyroid status is: {status}")
