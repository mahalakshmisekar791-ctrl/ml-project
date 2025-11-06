import  streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_pipeline():
    with open('final_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

def main():
    st.title("Credit Card Fraud Detection")

    pipeline = load_pipeline()

    # Try to get the pipeline's expected feature names
    try:
        expected_features = list(pipeline.named_steps['scaler'].feature_names_in_)
    except AttributeError:
        # Fall back: Specify manually if above fails
        expected_features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7',
                             'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
                             'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
                             'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

    st.sidebar.header("Input Transaction Data")

    input_data = {}
    for feature in expected_features:
        input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

    input_df = pd.DataFrame([input_data])

    # Reindex so input columns exactly match expected features
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    # Predict using pipeline
    prediction = pipeline.predict(input_df)[0]
    pred_proba = pipeline.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.write("Legitimate" if prediction == 0 else "Fraudulent")

    st.subheader("Prediction Probability")
    st.write(f"Fraud Probability: {pred_proba:.4f}")

if __name__ == "__main__":
    main()