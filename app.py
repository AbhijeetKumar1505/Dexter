import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import re
import requests
import warnings
from typing import Optional

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

# Title
st.title("📊 Interactive Data Analysis Dashboard")

# Initialize session states
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Sidebar for file upload
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

def safe_numeric_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns to float, handling errors gracefully."""
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass
    return df

def get_numeric_columns(df: pd.DataFrame) -> list:
    """Get list of numeric columns from dataframe."""
    return df.select_dtypes(include=[np.number]).columns.tolist()

# Load and preprocess data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = safe_numeric_conversion(df)
        st.session_state.data = df.copy()
        
        # Display raw data
        st.subheader("Raw Data")
        st.write(df)
        
        # Data preprocessing
        st.subheader("Data Preprocessing")
        
        # Handle missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            st.warning(f"Dataset contains {missing_values} missing values")
            missing_strategy = st.selectbox(
                "How to handle missing values?",
                ["Drop rows", "Fill with mean", "Fill with median", "Keep as is"]
            )
            
            if missing_strategy == "Drop rows":
                df = df.dropna()
            elif missing_strategy == "Fill with mean":
                df = df.fillna(df.mean())
            elif missing_strategy == "Fill with median":
                df = df.fillna(df.median())
        
        # Normalization option
        if st.checkbox("Normalize Data"):
            numeric_cols = get_numeric_columns(df)
            if numeric_cols:
                df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        
        st.session_state.processed_data = df.copy()
        st.write("Processed Data:")
        st.write(df)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        numeric_cols = get_numeric_columns(df)
        if numeric_cols:
            st.write(df[numeric_cols].describe())
        else:
            st.warning("No numeric columns found for summary statistics")
        
        # Statistical Analysis
        st.subheader("Statistical Analysis")
        if st.checkbox("Perform Hypothesis Testing (T-test)"):
            numeric_cols = get_numeric_columns(df)
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    column1 = st.selectbox("Select first column for T-test", numeric_cols)
                with col2:
                    column2 = st.selectbox("Select second column for T-test", numeric_cols)
                
                if not df[column1].empty and not df[column2].empty:
                    t_stat, p_value = stats.ttest_ind(
                        df[column1].dropna(),
                        df[column2].dropna()
                    )
                    st.write(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
                else:
                    st.error("Selected columns contain no valid data for T-test")
            else:
                st.warning("Need at least 2 numeric columns for T-test")
        
        if st.checkbox("Generate Correlation Matrix"):
            numeric_cols = get_numeric_columns(df)
            if numeric_cols:
                corr_matrix = df[numeric_cols].corr()
                st.write("Correlation Matrix:")
                st.write(corr_matrix)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                plt.title("Correlation Heatmap")
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("No numeric columns found for correlation analysis")
        
        # Data Visualization
        st.subheader("Data Visualization")
        numeric_cols = get_numeric_columns(df)
        
        if numeric_cols:
            plot_type = st.selectbox(
                "Select Plot Type",
                ["Scatter Plot", "Bar Chart", "Histogram", "Box Plot", "Violin Plot"]
            )
            
            try:
                if plot_type == "Scatter Plot":
                    x_axis = st.selectbox("Select X-axis", numeric_cols)
                    y_axis = st.selectbox("Select Y-axis", numeric_cols)
                    fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
                    st.plotly_chart(fig)
                
                elif plot_type == "Bar Chart":
                    x_axis = st.selectbox("Select X-axis", df.columns)
                    y_axis = st.selectbox("Select Y-axis", numeric_cols)
                    fig = px.bar(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
                    st.plotly_chart(fig)
                
                elif plot_type == "Histogram":
                    column = st.selectbox("Select Column", numeric_cols)
                    fig = px.histogram(df, x=column, title=f"Distribution of {column}")
                    st.plotly_chart(fig)
                
                elif plot_type in ["Box Plot", "Violin Plot"]:
                    y_axis = st.selectbox("Select Column", numeric_cols)
                    x_axis = st.selectbox("Select Grouping Column (optional)", ["None"] + df.columns.tolist())
                    
                    if x_axis == "None":
                        if plot_type == "Box Plot":
                            fig = px.box(df, y=y_axis)
                        else:
                            fig = px.violin(df, y=y_axis)
                    else:
                        if plot_type == "Box Plot":
                            fig = px.box(df, x=x_axis, y=y_axis)
                        else:
                            fig = px.violin(df, x=x_axis, y=y_axis)
                    
                    st.plotly_chart(fig)
            
            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")
        else:
            st.warning("No numeric columns available for visualization")
        
        # Predictive Modeling
        st.subheader("Predictive Modeling")
        if st.checkbox("Train a Model"):
            numeric_cols = get_numeric_columns(df)
            if len(numeric_cols) >= 2:
                target = st.selectbox("Select Target Variable", numeric_cols)
                features = st.multiselect(
                    "Select Features",
                    [col for col in numeric_cols if col != target],
                    default=[col for col in numeric_cols if col != target][:3]
                )
                
                if features:
                    try:
                        X = df[features].dropna()
                        y = df[target].dropna()
                        
                        # Ensure X and y have the same length
                        common_index = X.index.intersection(y.index)
                        X = X.loc[common_index]
                        y = y.loc[common_index]
                        
                        if len(X) > 0:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            model_type = st.selectbox(
                                "Select Model",
                                ["Linear Regression", "Decision Tree", "Random Forest"]
                            )
                            
                            if model_type == "Linear Regression":
                                model = LinearRegression()
                            elif model_type == "Decision Tree":
                                model = DecisionTreeRegressor(random_state=42)
                            else:
                                model = RandomForestRegressor(random_state=42)
                            
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            st.write("Model Performance:")
                            st.write(f"Mean Squared Error: {mse:.4f}")
                            st.write(f"R-squared: {r2:.4f}")
                            
                            if st.checkbox("Make Predictions"):
                                input_data = {}
                                for feature in features:
                                    input_data[feature] = st.number_input(
                                        f"Enter {feature}",
                                        value=float(df[feature].mean()),
                                        format="%.2f"
                                    )
                                input_df = pd.DataFrame([input_data])
                                prediction = model.predict(input_df)
                                st.write(f"Predicted {target}: {prediction[0]:.2f}")
                        else:
                            st.error("Not enough valid data points for modeling")
                    except Exception as e:
                        st.error(f"Error in model training: {str(e)}")
            else:
                st.warning("Need at least 2 numeric columns for modeling")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a CSV file to get started.") 