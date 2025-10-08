import streamlit as st
import requests
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="UPI Shield: Fraud Detection",
    page_icon="🛡️",
    layout="centered"
)

# --- App Title and Description ---
st.title("🛡️ UPI Shield: Real-Time Fraud Detection")
st.write(
    "Enter transaction details below to get a real-time fraud prediction. "
    "This interface communicates with a backend FastAPI model to analyze the data."
)

# --- Sidebar for Input Fields ---
st.sidebar.header("Enter Transaction Features")

# To make it user-friendly, we'll use a mix of sliders and number inputs.
# The default values are from the legitimate transaction sample.
def user_input_features():
    time = st.sidebar.number_input('Transaction Time (seconds)', value=0.0)
    amount = st.sidebar.number_input('Transaction Amount', value=149.62)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Feature values (V1-V28)")
    st.sidebar.markdown(
        "<p style='font-size: 12px; color: grey;'>"
        "These are anonymized features from the dataset. "
        "You can use the defaults or change them to test the model."
        "</p>", unsafe_allow_html=True
    )
    
    # Create sliders for all V features
    v_features = {}
    default_values = {
        'V1': -1.359, 'V2': -0.072, 'V3': 2.536, 'V4': 1.378, 'V5': -0.338,
        'V6': 0.462, 'V7': 0.239, 'V8': 0.098, 'V9': 0.363, 'V10': 0.09,
        'V11': -0.551, 'V12': -0.617, 'V13': -0.991, 'V14': -0.311, 'V15': 1.468,
        'V16': -0.47, 'V17': 0.207, 'V18': 0.025, 'V19': 0.403, 'V20': 0.251,
        'V21': -0.018, 'V22': 0.277, 'V23': -0.11, 'V24': 0.066, 'V25': 0.128,
        'V26': -0.189, 'V27': 0.133, 'V28': -0.021
    }

    for i in range(1, 29):
        feature_name = f'V{i}'
        v_features[feature_name] = st.sidebar.slider(
            feature_name, -50.0, 50.0, default_values[feature_name], 0.01
        )

    data = {
        'Time': time,
        'Amount': amount,
        **v_features # Unpack the v_features dictionary
    }
    
    # Reorder columns to match the model's training order
    column_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    features_df = pd.DataFrame(data, index=[0])
    features_df = features_df[column_order]

    return features_df, data

input_df, input_data_dict = user_input_features()

# Display the user input in the main area
st.subheader("Transaction Data to be Analyzed:")
st.dataframe(input_df)

# --- Prediction Logic ---
if st.button("Analyze Transaction"):
    # API endpoint URL
    api_url = "http://127.0.0.1:8000/predict"
    
    with st.spinner('Sending data to the model...'):
        try:
            # Send POST request to the FastAPI endpoint
            response = requests.post(api_url, json=input_data_dict)
            
            # Check if the request was successful
            if response.status_code == 200:
                prediction_data = response.json()
                prediction = prediction_data['prediction']
                score = float(prediction_data['fraud_probability_score'])
                
                st.success("Analysis Complete!")
                st.subheader("Prediction Result:")
                
                if prediction == "Fraud":
                    st.error(f"Prediction: **{prediction}**")
                    st.metric(label="Fraud Probability Score", value=f"{score:.4f}")
                    st.warning("This transaction is highly likely to be fraudulent. Please take action.", icon="⚠️")
                else:
                    st.success(f"Prediction: **{prediction}**")
                    st.metric(label="Fraud Probability Score", value=f"{score:.4f}")
                    st.info("This transaction appears to be legitimate.", icon="✅")

            else:
                st.error(f"Error from API: Status code {response.status_code}")
                st.json(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the API. Please ensure the FastAPI server is running.")
            st.error(str(e))