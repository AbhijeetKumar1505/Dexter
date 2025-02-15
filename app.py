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
import os
from huggingface_hub import login

# Hugging Face Login
def huggingface_login():
    try:
        # Check if Hugging Face token is already set
        if "HUGGINGFACE_TOKEN" not in os.environ:
            # Prompt user to enter Hugging Face token
            st.sidebar.subheader("Hugging Face Login")
            hf_token = st.sidebar.text_input("Enter your Hugging Face token (get it from https://huggingface.co/settings/tokens):", type="password")
            if hf_token:
                # Log in to Hugging Face
                login(token=hf_token)
                st.sidebar.success("Logged in to Hugging Face successfully!")
                os.environ["HUGGINGFACE_TOKEN"] = hf_token
            else:
                st.sidebar.warning("Please enter your Hugging Face token to access models.")
    except Exception as e:
        st.sidebar.error(f"Failed to log in to Hugging Face: {e}")

# Call Hugging Face login function
huggingface_login()

# Load DeepSeek model (replace with the actual model name)
if "HUGGINGFACE_TOKEN" in os.environ:
    from transformers import pipeline
    nlp = pipeline("text-generation", model="deepseek-ai/deepseek-math-7b")  # Example model
else:
    nlp = None

# Page configuration
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

# Title
st.title("📊 Interactive Data Analysis Dashboard")

# Sidebar for file upload
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state.data = None

# Load and preprocess data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.data = df

    # Display raw data
    st.subheader("Raw Data")
    st.write(df)

    # Data preprocessing
    st.subheader("Data Preprocessing")
    if st.checkbox("Handle Missing Values"):
        df = df.dropna()  # Drop missing values
    if st.checkbox("Normalize Data"):
        df = (df - df.mean()) / df.std()  # Normalize data
    st.write("Processed Data:")
    st.write(df)

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Statistical Analysis
    st.subheader("Statistical Analysis")
    if st.checkbox("Perform Hypothesis Testing (T-test)"):
        col1, col2 = st.columns(2)
        with col1:
            column1 = st.selectbox("Select first column for T-test", df.columns)
        with col2:
            column2 = st.selectbox("Select second column for T-test", df.columns)
        t_stat, p_value = stats.ttest_ind(df[column1], df[column2])
        st.write(f"T-statistic: {t_stat}, P-value: {p_value}")

    if st.checkbox("Generate Correlation Matrix"):
        corr_matrix = df.corr()
        st.write("Correlation Matrix:")
        st.write(corr_matrix)
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, ax=ax)
        st.pyplot(fig)

    # Data Visualization
    st.subheader("Data Visualization")
    plot_type = st.selectbox("Select Plot Type", ["Scatter Plot", "Bar Chart", "Histogram", "Pair Plot", "Heatmap"])
    if plot_type == "Scatter Plot":
        x_axis = st.selectbox("Select X-axis", df.columns)
        y_axis = st.selectbox("Select Y-axis", df.columns)
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
        st.plotly_chart(fig)
    elif plot_type == "Bar Chart":
        x_axis = st.selectbox("Select X-axis", df.columns)
        y_axis = st.selectbox("Select Y-axis", df.columns)
        fig = px.bar(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
        st.plotly_chart(fig)
    elif plot_type == "Histogram":
        column = st.selectbox("Select Column", df.columns)
        fig = px.histogram(df, x=column, title=f"Distribution of {column}")
        st.plotly_chart(fig)
    elif plot_type == "Pair Plot":
        fig = sns.pairplot(df)
        st.pyplot(fig)
    elif plot_type == "Heatmap":
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot(fig)

    # Predictive Modeling
    st.subheader("Predictive Modeling")
    if st.checkbox("Train a Model"):
        target = st.selectbox("Select Target Variable", df.columns)
        features = st.multiselect("Select Features", df.columns.drop(target))
        if features:
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_type = st.selectbox("Select Model", ["Linear Regression", "Decision Tree", "Random Forest"])
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Decision Tree":
                model = DecisionTreeRegressor()
            elif model_type == "Random Forest":
                model = RandomForestRegressor()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("Model Performance:")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
            st.write(f"R-squared: {r2_score(y_test, y_pred)}")

            if st.checkbox("Make Predictions"):
                input_data = {}
                for feature in features:
                    input_data[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)
                st.write(f"Predicted {target}: {prediction[0]}")

    # Prompt Section with DeepSeek Model
    st.subheader("🔍 Ask Questions About Your Data")
    user_prompt = st.text_input("Enter your question (e.g., 'What fraction of households have only one two-wheeler?')")

    if user_prompt:
        try:
            if nlp:
                # Use DeepSeek model to interpret the question
                response = nlp(user_prompt)
                st.write("DeepSeek Model Response:")
                st.write(response)

                # Extract intent and entities from the response
                if "fraction" in user_prompt.lower():
                    # Use regex to extract the condition (e.g., "only one two-wheeler")
                    match = re.search(r"only one (\w+)", user_prompt.lower())
                    if match:
                        column_name = match.group(1)  # Extract column name (e.g., "two-wheeler")
                        if column_name in df.columns:
                            # Calculate the fraction of households with only one two-wheeler
                            fraction = (df[column_name] == 1).mean()
                            st.write(f"The fraction of households with only one {column_name} is: {fraction:.2f}")
                        else:
                            st.error(f"Column '{column_name}' not found in the dataset.")
                    else:
                        st.error("Could not parse the condition. Please specify a valid condition.")

                elif "average" in user_prompt.lower():
                    column = user_prompt.split()[-1]  # Extract column name
                    if column in df.columns:
                        avg_value = df[column].mean()
                        st.write(f"The average of {column} is: {avg_value}")
                    else:
                        st.error(f"Column '{column}' not found in the dataset.")

                elif "correlation" in user_prompt.lower():
                    cols = [word for word in user_prompt.split() if word in df.columns]
                    if len(cols) == 2:
                        corr_value = df[cols[0]].corr(df[cols[1]])
                        st.write(f"The correlation between {cols[0]} and {cols[1]} is: {corr_value}")
                    else:
                        st.error("Please specify two valid columns for correlation.")

                elif "distribution" in user_prompt.lower():
                    column = user_prompt.split()[-1]
                    if column in df.columns:
                        fig = px.histogram(df, x=column, title=f"Distribution of {column}")
                        st.plotly_chart(fig)
                    else:
                        st.error(f"Column '{column}' not found in the dataset.")

                else:
                    st.warning("Sorry, I don't understand that question. Try asking about fractions, averages, correlations, or distributions.")
            else:
                st.error("Hugging Face model is not loaded. Please log in with your Hugging Face token.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a CSV file to get started.")