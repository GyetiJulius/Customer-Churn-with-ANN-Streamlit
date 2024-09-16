import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import streamlit as st

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the pickle files
with open('geography_one_hot.pkl', 'rb') as file:
    geography_one_hot = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Streamlit web app
st.title('Customer Churn Model')

# Input Data
geography = st.selectbox('Geography', geography_one_hot.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Number of Products')
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# Prepare  the data
input_data = pd.DataFrame({
    'Age':[age],
    'Balance':[balance],
    'CreditScore':[credit_score],
    'EstimatedSalary':[estimated_salary],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'NumOfProducts':[num_of_products],
    'Tenure':[tenure]
})

# One hot encode 'Geography'
geo_encoded = geography_one_hot.transform([geography]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns = geography_one_hot.get_feature_names_out(['Geography']))

# Combine geography dataframe with the input dataframe
input_data = pd.concat([input_data.reset_index(drop = True), geo_df], axis = 1)

# Scale the input data
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Predict Churn
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

# Show preb prob
st.write(f"Prediction probability: {prediction_probability:.2f}")

# Output
if prediction_probability > 0.5:
    st.write('Customer is likely to churn')
else:
    st.write('Customer is not likely to churn')
