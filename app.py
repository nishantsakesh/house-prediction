import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

st.set_page_config(layout="wide", page_title="Boston Housing Price Predictor")

# Function to load data from the CSV file
@st.cache_data # Cache data to avoid reloading every time the app reruns
def load_data():
    # --- Yahan path update kiya gaya hai ---
    file_path = os.path.join('data', 'boston.csv')
    
    # Optional: Debugging lines to check the path
    # st.write(f"Current working directory: {os.getcwd()}")
    # st.write(f"Looking for file at: {os.path.abspath(file_path)}")
    
    if not os.path.exists(file_path):
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"An error occurred while loading the CSV file: {e}")
        return pd.DataFrame()


# Load the dataset
df = load_data()

if not df.empty:
    # Define features (X) and target (y)
    X = df.drop('MEDV', axis=1) # All columns except 'MEDV' are features
    y = df['MEDV'] # 'MEDV' is the target variable

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit scaler on training data and transform both training and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Streamlit App Title and Introduction
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 3.5em;
            color: #4CAF50;
            text-align: center;
            text-shadow: 2px 2px 4px #aaa;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }
        .sub-header {
            font-size: 1.8em;
            color: #333333;
            text-align: center;
            margin-bottom: 30px;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size:1.2rem;
            color: #4CAF50;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        }
        .section-title {
            color: #2E8B57; /* Darker green */
            font-size: 2em;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-top: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 class='main-header'>🏡 Boston Housing Price Predictor 📊</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Building an end-to-end Machine Learning project using Streamlit and the Boston Housing Dataset.</p>", unsafe_allow_html=True)

    st.write("---")

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Overview & Raw Data", "EDA", "Model Training", "Prediction"])

    with tab1:
        st.markdown("<h2 class='section-title'>Dataset Overview & Raw Data</h2>", unsafe_allow_html=True)
        st.write("This app predicts housing prices in Boston based on various features.")
        
        st.subheader("Raw Dataset (First 5 Rows)")
        st.dataframe(df.head())

        st.subheader("Dataset Description")
        st.write(df.describe())

        st.subheader("Missing Values Check")
        # Display missing values count and percentage
        missing_data = df.isnull().sum()
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        missing_df = pd.DataFrame({'Missing Count': missing_data, 'Percentage (%)': missing_percentage})
        st.dataframe(missing_df[missing_df['Missing Count'] > 0])
        if missing_df[missing_df['Missing Count'] > 0].empty:
            st.success("No missing values found in the dataset! 🎉")
        else:
            st.warning("Missing values found. (For this dataset, it's typically clean)")


    with tab2:
        st.markdown("<h2 class='section-title'>Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)

        st.subheader("Feature Distributions")
        # Display histograms for all features
        num_cols = len(df.columns) - 1 # Exclude target variable
        cols_per_row = 4
        rows_needed = int(np.ceil(num_cols / cols_per_row))

        for i in range(rows_needed):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < num_cols:
                    feature = df.columns[idx]
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.histplot(df[feature], kde=True, ax=ax, color='#4CAF50')
                        ax.set_title(f'Distribution of {feature}', fontsize=12)
                        ax.set_xlabel(feature, fontsize=10)
                        ax.set_ylabel('Frequency', fontsize=10)
                        plt.tight_layout()
                        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt=".2f", ax=ax, linewidths=.5)
        ax.set_title('Correlation Matrix of Features', fontsize=16)
        plt.tight_layout()
        st.pyplot(fig)
        st.info("The correlation heatmap shows the relationships between different features and the target variable (MEDV). Positive values indicate a positive correlation, negative values indicate a negative correlation, and values close to zero indicate a weak or no linear correlation.")


    with tab3:
        st.markdown("<h2 class='section-title'>Model Training</h2>", unsafe_allow_html=True)

        # Model Selection
        model_choice = st.selectbox(
            "Choose a Regression Model:",
            ("Linear Regression", "Ridge Regression", "Random Forest Regressor")
        )

        model = None
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Ridge Regression":
            alpha = st.slider("Select Alpha (Regularization Strength) for Ridge", 0.1, 10.0, 1.0)
            model = Ridge(alpha=alpha)
        elif model_choice == "Random Forest Regressor":
            n_estimators = st.slider("Select Number of Estimators for Random Forest", 50, 500, 100, step=50)
            max_depth = st.slider("Select Max Depth for Random Forest", 1, 20, 10)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        if st.button(f"Train {model_choice} Model"):            
            # Train the model
            model.fit(X_train_scaled, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test_scaled)

            # Evaluate the model
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            st.subheader(f"Model Performance: {model_choice}")
            st.write(f"**R-squared (R²):** {r2:.3f}")
            st.write(f"**Mean Absolute Error (MAE):** {mae:.3f}")
            st.write(f"**Mean Squared Error (MSE):** {mse:.3f}")

            # Store the trained model and scaler in session state for prediction tab
            st.session_state['trained_model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['features'] = X.columns.tolist()

            st.success(f"{model_choice} trained successfully! Check out the 'Prediction' tab to try it out.")



    with tab4:
        st.markdown("<h2 class='section-title'>Make a Prediction</h2>", unsafe_allow_html=True)

        if 'trained_model' in st.session_state and 'scaler' in st.session_state and 'features' in st.session_state:
            model = st.session_state['trained_model']
            scaler = st.session_state['scaler']
            features = st.session_state['features']

            st.write("Enter the values for the features to predict the housing price:")

            # Create input fields for each feature
            input_data = {}
            # Min/Max values for sliders (taken from the describe() output or typical ranges)
            feature_ranges = {
                'CRIM': (0.006, 89.0),
                'ZN': (0.0, 100.0),
                'INDUS': (0.46, 27.74),
                'CHAS': (0, 1),
                'NOX': (0.38, 0.87),
                'RM': (3.5, 8.8),
                'AGE': (2.9, 100.0),
                'DIS': (1.1, 12.1),
                'RAD': (1, 24),
                'TAX': (187.0, 711.0),
                'PTRATIO': (12.6, 22.0),
                'B': (0.32, 396.9),
                'LSTAT': (1.73, 37.97)
            }

            # Organize input fields in columns for better UI
            num_inputs_per_row = 3
            cols = st.columns(num_inputs_per_row)
            for i, feature in enumerate(features):
                with cols[i % num_inputs_per_row]:
                    min_val, max_val = feature_ranges.get(feature, (df[feature].min(), df[feature].max()))
                    default_val = df[feature].median() 

                    if feature in ['CHAS']: 
                        input_data[feature] = st.radio(
                            f"**{feature}** (Bounds: {min_val:.2f}-{max_val:.2f})",
                            options=[int(min_val), int(max_val)],
                            index=0 if default_val == min_val else 1, 
                            key=f"input_{feature}"
                        )
                    else:
                        input_data[feature] = st.slider(
                            f"**{feature}** (Bounds: {min_val:.2f}-{max_val:.2f})",
                            float(min_val), float(max_val), float(default_val),
                            key=f"input_{feature}"
                        )
            
            # Prediction button
            if st.button("Predict House Price", key="predict_button"):
                # Create a DataFrame from user input
                input_df = pd.DataFrame([input_data])
                
                # Scale the input data using the *trained* scaler
                input_scaled = scaler.transform(input_df[features])

                # Make prediction
                predicted_price = model.predict(input_scaled)[0]

                st.success(f"**Predicted House Price: ${predicted_price*1000:,.2f}**")
                st.info("The price is in thousands of dollars ($1000s).")

        else:
            st.warning("Please train a model in the 'Model Training' tab first to enable predictions.")

    st.write("---")
else:
    st.error("Application could not load the dataset. Please check the path and file integrity.")