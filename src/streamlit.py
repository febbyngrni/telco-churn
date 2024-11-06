import streamlit as st
import requests
from PIL import Image

st.title('Telecom Company Churn Prediction')

header_image = Image.open('assets/header_image.png')
st.image(header_image)

with st.form(key = 'telco_form'):
    tenure_months = st.number_input(
        label = 'Enter the number of months you have been a customer:',
        min_value = 0,
        max_value = 72,
        help = 'Value range from 0 to 72'
    )

    monthly_charges = st.number_input(
        label = 'Enter your monthly charge:',
        min_value = 18.25,
        max_value = 118.75,
        help = 'Value range from 18.25 to 118.75'
    )

    total_charges = st.number_input(
        label = 'Enter your total charges:',
        min_value = 0.0,
        max_value = 8684.8,
        help = 'Value range from 0 to 8684.8'
    )

    gender = st.selectbox(
        label = 'Select gender:',
        options = (
            'Female',
            'Male'
        )
    )

    senior_citizen = st.selectbox(
        label = 'Are you a senior citizen?',
        options = (
            'Yes',
            'No'
        )
    )

    partner = st.selectbox(
        label = 'Do you have a partner?',
        options = (
            'Yes',
            'No'
        )
    )

    dependents = st.selectbox(
        label = 'Do you any dependents?',
        options = (
            'Yes',
            'No'
        )
    )

    phone_service = st.selectbox(
        label = 'Do you subscribe to a phone service?',
        options = (
            'Yes',
            'No'
        )
    )

    multiple_lines = st.selectbox(
        label = 'Do you have multiple lines for your phone service?',
        options = (
            'Yes',
            'No',
            'No phone service'
        )
    )

    internet_service = st.selectbox(
        label = 'Select your internet service type:',
        options = (
            'Fiber optic',
            'DSL',
            'No'
        )
    )

    online_security = st.selectbox(
        label = 'Do you have an online security service?',
        options = (
            'Yes',
            'No',
            'No internet service'
        )
    )

    online_backup = st.selectbox(
        label = 'Do you have an online backup service?',
        options = (
            'Yes',
            'No',
            'No internet service'
        )
    )

    device_protection = st.selectbox(
        label = 'Do you have device protection?',
        options = (
            'Yes',
            'No',
            'No internet service'
        )
    )

    tech_support = st.selectbox(
        label = 'Do you subscribe to tech support?',
        options = (
            'Yes',
            'No',
            'No internet service'
        )
    )

    streaming_tv = st.selectbox(
        label = 'Do you subscribe to a TV streaming service?',
        options = (
            'Yes',
            'No',
            'No internet service'
        )
    )

    streaming_movies = st.selectbox(
        label = 'Do you subscribe to a movie streaming service?',
        options = (
            'Yes',
            'No',
            'No internet service'
        )
    )

    contract = st.selectbox(
        label = 'Select your contract type:',
        options = (
            'Two year',
            'One year',
            'Month-to-month'
        )
    )

    paperless_billing = st.selectbox(
        label = 'Do you use paperless billing?',
        options = (
            'Yes',
            'No'
        )
    )

    payment_method = st.selectbox(
        label='Select your payment method:',
        options=(
            'E-Check',
            'M-Check',
            'Auto Bank',
            'Auto Card'
        )
    )

    submitted = st.form_submit_button('Predict')

    if submitted:
        raw_data = {
            'tenure_months' : tenure_months,
            'monthly_charges' : monthly_charges,
            'total_charges' :total_charges,
            'gender' : gender,
            'senior_citizen' : senior_citizen,
            'partner' : partner,
            'dependents' : dependents,
            'phone_service' : phone_service,
            'multiple_lines' : multiple_lines,
            'internet_service' : internet_service,
            'online_security' : online_security,
            'online_backup' : online_backup,
            'device_protection' : device_protection,
            'tech_support' : tech_support,
            'streaming_tv' : streaming_tv,
            'streaming_movies' : streaming_movies,
            'contract' : contract,
            'paperless_billing' : paperless_billing,
            'payment_method' : payment_method
        }

        with st.spinner('Sending data to prediction server...'):
            res = requests.post('http://api_backend:8080/predict', json = raw_data).json()

        if res['error_msg'] != '':
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res['res'] == 'Yes':
                st.warning("Predicted Churn: YES")
            else:
                st.success("Predicted Churn: NO")