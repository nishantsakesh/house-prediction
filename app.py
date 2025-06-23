import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns          

try:
    df = pd.read_csv('boston.csv') 
    st.sidebar.success("Dataset loaded successfully for visualizations!") 
except FileNotFoundError:
    st.error("Error: 'boston.csv' not found. Visualizations might be limited.")
    df = None 

# Page configuration
st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed" 
)

# Title of the web app
st.title("🏠 Boston House Price Prediction App")
st.markdown("Enter the details of the house to predict its price in Boston.")

# Loading the trained model and scaler
try:
    model = joblib.load('house_price_prediction_model.pkl')
    scaler = joblib.load('house_price_scaler.pkl')
    st.success("Model and Scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or Scaler files not found. Please ensure 'house_price_prediction_model.pkl' and 'house_price_scaler.pkl' are in the same directory as this app.")
    st.stop() 

st.header("Enter House Characteristics")
st.markdown("Adjust the sliders and input fields below to set the house details.")


col1, col2 = st.columns(2) 

with col1:
    crim = st.slider("1. Crime Rate (CRIM)", 0.0, 90.0, 0.1, key='crim')
    zn = st.slider("2. Residential Land Zoned (ZN)", 0.0, 100.0, 0.0, key='zn')
    indus = st.slider("3. Non-retail Business Acres (INDUS)", 0.0, 30.0, 5.0, key='indus')
    chas = st.selectbox("4. Borders Charles River (CHAS)", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No", key='chas')
    nox = st.slider("5. Nitric Oxides Concentration (NOX)", 0.3, 1.0, 0.5, key='nox')
    rm = st.slider("6. Average Number of Rooms (RM)", 3.0, 9.0, 6.0, key='rm')

with col2:
    age = st.slider("7. Age of House (AGE)", 0.0, 100.0, 50.0, key='age')
    dis = st.slider("8. Distance to Employment Centers (DIS)", 1.0, 15.0, 4.0, key='dis')
    rad = st.slider("9. Radial Highways Access (RAD)", 1.0, 25.0, 5.0, key='rad')
    tax = st.slider("10. Property Tax Rate (TAX)", 180.0, 750.0, 300.0, key='tax')
    ptratio = st.slider("11. Pupil-Teacher Ratio (PTRATIO)", 10.0, 23.0, 18.0, key='ptratio')
    b = st.slider("12. Black Population Proportion (B)", 0.0, 400.0, 350.0, key='b')
    lstat = st.slider("13. Lower Status Population (LSTAT)", 1.0, 40.0, 10.0, key='lstat')

# Create a dictionary from the input values
input_data = {
    'CRIM': crim,
    'ZN': zn,
    'INDUS': indus,
    'CHAS': chas,
    'NOX': nox,
    'RM': rm,
    'AGE': age,
    'DIS': dis,
    'RAD': rad,
    'TAX': tax,
    'PTRATIO': ptratio,
    'B': b,
    'LSTAT': lstat
}

# Convert the dictionary to a pandas DataFrame
input_df = pd.DataFrame([input_data])

st.subheader("Entered House Details:")
st.write(input_df)

# Scale the input data using the loaded scaler
try:
    # Ensure the order of columns in input_df matches the order during model training
    scaled_input_data = scaler.transform(input_df)

    # Make prediction button
    if st.button("Predict House Price"):
        prediction = model.predict(scaled_input_data)
        st.success(f"The predicted house price (MEDV) is: **${prediction[0]:,.2f}**")

except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
    st.warning("Please check if the input values are valid and if the model/scaler are loaded correctly.")



# Visualizations Section 
st.markdown("---")
st.header("📊 Data Insights")

if df is not None: 
    # 1. Distribution of MEDV
    st.subheader("1. Distribution of House Prices (MEDV)")
    fig_medv, ax_medv = plt.subplots(figsize=(5, 5))
    sns.histplot(df['MEDV'], kde=True, ax=ax_medv)
    ax_medv.set_title('Distribution of Median House Prices')
    ax_medv.set_xlabel('Median Value of Owner-Occupied Homes ($1000s)')
    ax_medv.set_ylabel('Frequency')
    st.pyplot(fig_medv) 

    # 2. Relationship between RM and MEDV
    st.subheader("2. Average Rooms (RM) vs. House Price")
    fig_rm, ax_rm = plt.subplots(figsize=(5, 5))
    sns.scatterplot(x='RM', y='MEDV', data=df, ax=ax_rm)
    ax_rm.set_title('Average Number of Rooms vs. House Price')
    ax_rm.set_xlabel('Average number of rooms (RM)')
    ax_rm.set_ylabel('Median Value ($1000s)')
    st.pyplot(fig_rm)

    # 3. Relationship between LSTAT and MEDV
    st.subheader("3. Lower Status Population (LSTAT) vs. House Price")
    fig_lstat, ax_lstat = plt.subplots(figsize=(5, 5))
    sns.scatterplot(x='LSTAT', y='MEDV', data=df, ax=ax_lstat)
    ax_lstat.set_title('Lower Status Population vs. House Price')
    ax_lstat.set_xlabel('% lower status of the population (LSTAT)')
    ax_lstat.set_ylabel('Median Value ($1000s)')
    st.pyplot(fig_lstat)

    # Optional: Correlation Heatmap
    st.subheader("4. Feature Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(5, 5))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', fmt=".2f", ax=ax_corr) 
    ax_corr.set_title('Correlation Matrix of Features')
    st.pyplot(fig_corr)

else:
    st.warning("Data not loaded, unable to display visualizations.")



st.markdown("---")
st.markdown("Developed by Nishant Sakesh | Data Source: Boston Housing Dataset")