# Fraud Detection Project

# Overview

This project aims to detect fraudulent transactions using machine learning techniques. The dataset used for this project contains information about transactions, including the type of transaction, amount, and balance before and after the transaction.

# Project Structure

- Data Preprocessing: Handling missing values, encoding categorical variables, and scaling numerical variables.
- Exploratory Data Analysis (EDA): Visualizing transaction types, fraud rates, and correlations between variables.
- Modeling: Using logistic regression with a pipeline to classify transactions as fraudulent or not.
- Evaluation: Assessing the model's performance using classification reports and confusion matrices.

# Key Features

- _Data Preprocessing_: Handling missing values, encoding categorical variables, and scaling numerical variables using `StandardScaler` and `OneHotEncoder`.
- _Feature Engineering_: Creating new features such as balance differences between origin and destination accounts.
- _Modeling_: Using logistic regression with a pipeline to classify transactions as fraudulent or not.
- _Model Evaluation_: Assessing the model's performance using classification reports and confusion matrices.

# Dependencies

- `pandas` for data manipulation
- `numpy` for numerical computations
- `matplotlib` and `seaborn` for data visualization
- `scikit-learn` for machine learning
- `joblib` for saving the trained model
- `streamlit` for building the web application

# Usage

1. Clone the repository: `git clone https://github.com/your-username/Fraud-Detection.git`
2. Install the dependencies: `pip install -r requirements.txt`
3. Run the script: `streamlit run fraud_detection_app.py`

# Streamlit App

The Streamlit app allows users to input transaction details and predict whether the transaction is likely to be fraudulent or not.

```
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("sample_fraud_detection_pipeline.pkl")

# Create the Streamlit app
st.title("Fraud Detection Prediction App")
st.markdown("Please enter the transaction details and use the predict button")
st.divider()

# Input fields for transaction details
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH OUT", "DEPOSIT"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

# Predict button
if st.button("Predict"):
    # Create a dataframe with the input data
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    # Make a prediction
    prediction = model.predict(input_data)[0]

    # Display the prediction
    st.subheader(f"Prediction: '{int(prediction)}'")
    if prediction == 1:
        st.error("This is likely a FRAUD!!!")
    else:
        st.success("Transaction seems clean")
```

# Model Performance

The model's performance is evaluated using a classification report and a confusion matrix. The accuracy of the model is also calculated and printed.

# Future Work

- Experiment with different machine learning algorithms and techniques to improve the model's performance.
- Use more advanced feature engineering techniques to extract more relevant features from the data.
- Deploy the model in a production-ready environment using a framework like Streamlit or Flask.

# Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

# License

This project is licensed under the MIT License. See the LICENSE file for details.
