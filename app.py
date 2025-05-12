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
import json
from together import Together
import missingno as msno

# Initialize Together AI client with API key from secrets
together = Together(api_key=st.secrets["TOGETHER_API_KEY"])

def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data type issues to make DataFrame compatible with Arrow for Streamlit display
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_fixed = df.copy()
    
    # First, fix column names - replace unnamed columns with better names
    renamed_cols = {}
    for col in df_fixed.columns:
        if 'Unnamed:' in str(col):
            renamed_cols[col] = f"Column_{str(col).split(':')[1].strip()}"
    
    if renamed_cols:
        df_fixed = df_fixed.rename(columns=renamed_cols)
    
    # Check for columns with mixed types and convert to strings where needed
    for col in df_fixed.columns:
        # Check if the column has mixed types
        if df_fixed[col].dtype == 'object':
            # Try to convert to numeric if possible
            try:
                pd.to_numeric(df_fixed[col])
                df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
            except:
                # If not numeric, convert bytes to strings to avoid Arrow errors
                try:
                    # Convert any bytes objects to strings
                    mask = df_fixed[col].apply(lambda x: isinstance(x, bytes))
                    if mask.any():
                        df_fixed.loc[mask, col] = df_fixed.loc[mask, col].apply(
                            lambda x: x.decode('utf-8', errors='replace')
                        )
                except:
                    # Last resort: convert entire column to string
                    df_fixed[col] = df_fixed[col].astype(str)
    
    return df_fixed

def load_data(uploaded_file) -> pd.DataFrame:
    """
    Load data from uploaded file (CSV or Excel)
    """
    try:
        # Debug information
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"File type: {uploaded_file.type}")
        st.write(f"File size: {uploaded_file.size} bytes")
        
        # Create a copy of the file buffer to avoid issues with buffer position
        # This is needed because Streamlit might consume the buffer during upload
        file_content = uploaded_file.getvalue()
        
        file_type = uploaded_file.name.split('.')[-1].lower()
        st.write(f"Detected file extension: {file_type}")
        
        if file_type == 'csv':
            try:
                # Try pandas read_csv with BytesIO
                import io
                df = pd.read_csv(io.BytesIO(file_content))
                st.success(f"Successfully read CSV file")
                
                # Fix data types to ensure Arrow compatibility
                df = fix_data_types(df)
                
                return df
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                # Try a different approach with direct read
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success("Successfully read CSV with direct approach")
                    
                    # Fix data types to ensure Arrow compatibility
                    df = fix_data_types(df)
                    
                    return df
                except Exception as e2:
                    st.error(f"Error with direct CSV read: {str(e2)}")
            
        elif file_type in ['xlsx', 'xls']:
            try:
                # Try pandas read_excel with BytesIO
                import io
                df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
                st.success("Successfully read Excel file")
                
                # Fix data types to ensure Arrow compatibility
                df = fix_data_types(df)
                
                return df
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                # Try alternative approach
                try:
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                    st.success("Successfully read Excel with direct approach")
                    
                    # Fix data types to ensure Arrow compatibility
                    df = fix_data_types(df)
                    
                    return df
                except Exception as e2:
                    st.error(f"Error with direct Excel read: {str(e2)}")
                    
                    # If openpyxl fails, try xlrd for .xls files
                    if file_type == 'xls':
                        try:
                            df = pd.read_excel(io.BytesIO(file_content), engine='xlrd')
                            st.success("Successfully read XLS file with xlrd engine")
                            
                            # Fix data types to ensure Arrow compatibility
                            df = fix_data_types(df)
                            
                            return df
                        except Exception as e3:
                            st.error(f"Error with xlrd engine: {str(e3)}")
        else:
            raise Exception(f"Unsupported file format: {file_type}")
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Add a simple test function for file upload
def test_file_upload(uploaded_file):
    """Simple test function to check file upload functionality"""
    st.write("Testing file upload...")
    
    if uploaded_file is None:
        st.error("No file was uploaded")
        return
        
    try:
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"File type: {uploaded_file.type}")
        st.write(f"File size: {uploaded_file.size} bytes")
        
        # Try to read basic content
        content_preview = uploaded_file.read(1024)  # Read first 1KB
        if content_preview:
            st.success("Successfully read file content")
            st.write("Content preview (first 100 bytes):")
            st.code(str(content_preview[:100]))
            
            # Reset the file pointer for further processing
            uploaded_file.seek(0)
        else:
            st.warning("File appears to be empty")
            
    except Exception as e:
        st.error(f"Error in test function: {str(e)}")

# Function to generate data insights using Together AI
def generate_data_insights(question: str, df: pd.DataFrame) -> str:
    """
    Generate insights about the data using Together AI API
    """
    if not question or df is None or df.empty:
        return "Please provide a valid question and dataset."
        
    try:
        # Convert dataframe info to string
        df_info = f"""
        Dataset Info:
        - Shape: {df.shape}
        - Columns: {', '.join(df.columns)}
        - Data Types: {df.dtypes.to_dict()}
        - Summary Statistics: {df.describe().to_dict()}
        """
        
        # Construct the prompt
        prompt = f"""
        Given the following dataset information:
        {df_info}
        
        Question: {question}
        
        Please provide a detailed analysis and answer based on the data provided.
        """
        
        try:
            # Call Together AI API with a supported model
            response = together.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Using Mixtral model which is supported
                messages=[
                    {"role": "system", "content": "You are a helpful data analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                top_k=50
            )
            
            # Extract the generated text from the response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                return "No response generated from the API."
                
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return f"Error calling Together AI API: {str(e)}\n\nPlease check your API key and internet connection."
    except Exception as e:
        return f"Error generating insights: {str(e)}"

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
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

# Add a debug mode toggle
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=True)

# If debug mode is enabled, run the test function first
if debug_mode and uploaded_file is not None:
    st.subheader("⚙️ File Upload Debug Information")
    test_file_upload(uploaded_file)
    
    # Reset file pointer after testing
    if uploaded_file is not None:
        uploaded_file.seek(0)

def safe_numeric_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns to float, handling errors gracefully."""
    if df is None:
        return None
        
    df_copy = df.copy()
    try:
        # Get columns that look like they should be numeric
        potential_numeric = []
        for col in df_copy.columns:
            # Skip columns that are already numeric
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                continue
                
            # Check if column contains potential numeric values
            non_na_values = df_copy[col].dropna()
            if len(non_na_values) > 0:
                # Check a sample value
                sample = non_na_values.iloc[0]
                if isinstance(sample, str):
                    # Try to detect if it looks like a number
                    # Remove common formatting characters
                    clean_sample = re.sub(r'[$,\s%]', '', sample)
                    try:
                        float(clean_sample)
                        potential_numeric.append(col)
                    except (ValueError, TypeError):
                        continue
        
        # Convert potential numeric columns
        for col in potential_numeric:
            # Remove common formatting characters
            df_copy[col] = df_copy[col].astype(str).str.replace(r'[$,\s%]', '', regex=True)
            # Convert to numeric with NaN for errors
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            
        # Finally, ensure all object columns don't have mixed types
        df_copy = fix_data_types(df_copy)
            
        return df_copy
    except Exception as e:
        st.error(f"Error in numeric conversion: {str(e)}")
        return df

def analyze_missing_values(df: pd.DataFrame) -> dict:
    """
    Analyze missing values in the dataset
    """
    try:
        total_missing = df.isna().sum().sum()
        missing_by_column = df.isna().sum().to_dict()
        missing_percent = (df.isna().sum() / len(df) * 100).to_dict()
        
        return {
            "total_missing": total_missing,
            "missing_by_column": missing_by_column,
            "missing_percent": missing_percent
        }
    except Exception as e:
        st.error(f"Error analyzing missing values: {str(e)}")
        return {
            "total_missing": 0,
            "missing_by_column": {},
            "missing_percent": {}
        }

def handle_missing_values(df: pd.DataFrame, strategy: str = "none") -> pd.DataFrame:
    """
    Handle missing values based on selected strategy
    """
    if df is None:
        st.error("No data to process")
        return df
    
    df_copy = df.copy()
    
    try:
        if strategy == "drop_rows":
            # Drop rows with any missing values
            df_copy = df_copy.dropna()
            
        elif strategy == "drop_columns":
            # Drop columns with missing values above threshold (50%)
            threshold = len(df_copy) * 0.5
            df_copy = df_copy.dropna(axis=1, thresh=threshold)
            
        elif strategy == "fill_mean":
            # Fill numeric missing values with mean
            numeric_cols = df_copy.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                if df_copy[col].isna().any():  # Only process if has NAs
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
                    
        elif strategy == "fill_median":
            # Fill numeric missing values with median
            numeric_cols = df_copy.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                if df_copy[col].isna().any():  # Only process if has NAs
                    df_copy[col] = df_copy[col].fillna(df_copy[col].median())
                    
        elif strategy == "fill_mode":
            # Fill all missing values with mode
            for col in df_copy.columns:
                if df_copy[col].isna().any():  # Only process if has NAs
                    mode_value = df_copy[col].mode()
                    if not mode_value.empty:
                        df_copy[col] = df_copy[col].fillna(mode_value[0])
                        
        elif strategy == "fill_zeros":
            # Fill numeric missing values with zeros
            numeric_cols = df_copy.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                if df_copy[col].isna().any():  # Only process if has NAs
                    df_copy[col] = df_copy[col].fillna(0)
                    
        return df_copy
        
    except Exception as e:
        st.error(f"Error handling missing values: {str(e)}")
        return df  # Return original df on error

def visualize_missing_values(df: pd.DataFrame):
    """
    Create visualizations for missing values
    """
    try:
        # Missing value matrix plot
        st.subheader("Missing Value Matrix")
        fig_matrix = plt.figure(figsize=(10, 6))
        msno.matrix(df)
        plt.tight_layout()  # Ensure the plot fits well
        st.pyplot(fig_matrix)
        plt.close(fig_matrix)  # Close the figure to prevent warnings
        
        # Missing value correlation heatmap
        if df.isna().sum().sum() > 0:  # Only show if there are missing values
            try:
                st.subheader("Missing Value Correlation")
                fig_heatmap = plt.figure(figsize=(10, 8))
                msno.heatmap(df)
                plt.tight_layout()  # Ensure the plot fits well
                st.pyplot(fig_heatmap)
                plt.close(fig_heatmap)  # Close the figure to prevent warnings
            except Exception as e:
                st.warning(f"Could not generate missing value heatmap: {str(e)}")
                st.info("This usually happens when there are not enough missing values for correlation analysis.")
            
        # Missing value bar chart 
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Values': df.isna().sum().values,
            'Percentage': (df.isna().sum() / len(df) * 100).values
        }).sort_values('Missing Values', ascending=False)
        
        if missing_data['Missing Values'].sum() > 0:
            st.subheader("Missing Values by Column")
            try:
                fig = px.bar(
                    missing_data, 
                    x='Column', 
                    y='Missing Values',
                    hover_data=['Percentage'],
                    labels={'Percentage': 'Missing Percentage (%)'},
                    color='Percentage',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                fig.update_layout(xaxis_title="Column", yaxis_title="Missing Value Count")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate missing value bar chart: {str(e)}")
                # Fallback to simple table
                safe_display_dataframe(missing_data)
    except Exception as e:
        st.error(f"Error in visualizing missing values: {str(e)}")
        st.info("Falling back to summary statistics...")
        # Fallback to a simple summary table 
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Values': df.isna().sum().values,
            'Percentage': (df.isna().sum() / len(df) * 100).values
        }).sort_values('Missing Values', ascending=False)
        safe_display_dataframe(missing_data)

def get_numeric_columns(df: pd.DataFrame) -> list:
    """Get list of numeric columns from dataframe."""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def safe_display_dataframe(df, max_rows=10):
    """
    Safely display a DataFrame in Streamlit, handling Arrow conversion issues
    """
    try:
        if df is not None and not df.empty:
            # First try to display as is
            return st.dataframe(df.head(max_rows), use_container_width=True)
    except Exception as e:
        try:
            # If failed, try to fix data types
            df_fixed = fix_data_types(df)
            return st.dataframe(df_fixed.head(max_rows), use_container_width=True)
        except Exception as e2:
            # Last resort: convert to string representation
            st.error(f"Error displaying dataframe: {str(e2)}")
            st.code(df.head(max_rows).to_string())
            return None

# Load and preprocess data
if uploaded_file is not None:
    try:
        st.subheader("📄 Data Loading")
        df = load_data(uploaded_file)
        
        if df is not None:
            st.write("DataFrame loaded successfully!")
            st.write("Shape:", df.shape)
            st.write("Columns:", df.columns.tolist())
            
            # Data preview 
            st.subheader("Data Preview")
            safe_display_dataframe(df)
            
            # Convert to numeric where possible
            df = safe_numeric_conversion(df)
            st.session_state.data = df.copy()
            
            # Display raw data
            st.subheader("Raw Data")
            safe_display_dataframe(df)
            
            # Analyze missing values
            missing_values = analyze_missing_values(df)
            total_missing = missing_values["total_missing"]
            
            if total_missing > 0:
                st.subheader("Missing Values Analysis")
                st.info(f"📊 Dataset contains {total_missing} missing values")
                
                # Create tabs for different missing value views
                missing_tabs = st.tabs(["Summary", "Visualization", "Treatment"])
                
                with missing_tabs[0]:
                    # Summary statistics
                    missing_by_col = pd.DataFrame({
                        'Column': missing_values["missing_by_column"].keys(),
                        'Missing Count': missing_values["missing_by_column"].values(),
                        'Missing Percentage': [f"{p:.2f}%" for p in missing_values["missing_percent"].values()]
                    }).sort_values('Missing Count', ascending=False)
                    
                    safe_display_dataframe(missing_by_col)
                
                with missing_tabs[1]:
                    # Visualizations
                    visualize_missing_values(df)
                
                with missing_tabs[2]:
                    # Missing value treatment
                    st.write("Select a strategy to handle missing values:")
                    
                    missing_strategy = st.selectbox(
                        "Missing Value Treatment Strategy",
                        options=[
                            "none", 
                            "drop_rows", 
                            "drop_columns", 
                            "fill_mean", 
                            "fill_median", 
                            "fill_mode", 
                            "fill_zeros"
                        ],
                        format_func=lambda x: {
                            "none": "No treatment (keep missing values)",
                            "drop_rows": "Drop rows with missing values",
                            "drop_columns": "Drop columns with >50% missing values",
                            "fill_mean": "Fill numeric missing values with mean",
                            "fill_median": "Fill numeric missing values with median",
                            "fill_mode": "Fill all missing values with mode",
                            "fill_zeros": "Fill numeric missing values with zeros"
                        }.get(x, x)
                    )
                    
                    if missing_strategy != "none":
                        if st.button("Apply Treatment"):
                            with st.spinner("Applying missing value treatment..."):
                                try:
                                    treated_df = handle_missing_values(df, missing_strategy)
                                    
                                    if treated_df is not None:
                                        # Display results
                                        treated_missing = treated_df.isna().sum().sum()
                                        
                                        # Show before/after comparison
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric(
                                                label="Original Missing Values", 
                                                value=total_missing,
                                                delta=None
                                            )
                                        with col2:
                                            st.metric(
                                                label="Remaining Missing Values", 
                                                value=treated_missing,
                                                delta=-(total_missing - treated_missing),
                                                delta_color="inverse"
                                            )
                                        
                                        if treated_missing == 0:
                                            st.success("✅ All missing values have been handled!")
                                        else:
                                            st.warning(f"⚠️ There are still {treated_missing} missing values remaining.")
                                        
                                        # Update the dataframe
                                        df = treated_df
                                        st.session_state.data = df.copy()
                                        
                                        # Show the updated data
                                        st.subheader("Updated Data Preview")
                                        safe_display_dataframe(df)
                                    else:
                                        st.error("Treatment failed. The data was not modified.")
                                except Exception as e:
                                    st.error(f"Error during treatment: {str(e)}")
                    else:
                        st.info("Select a treatment strategy to handle missing values.")
            
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
            safe_display_dataframe(df)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            
            stats_tab1, stats_tab2 = st.tabs(["Describe", "Correlation"])
            with stats_tab1:
                st.write("Descriptive statistics:")
                desc_stats = df.describe(include='all').T
                desc_stats = desc_stats.round(2)  # Round to 2 decimal places
                safe_display_dataframe(desc_stats, max_rows=None)
            
            with stats_tab2:
                st.write("Correlation matrix:")
                numeric_cols = get_numeric_columns(df)
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols].corr().round(2)
                    safe_display_dataframe(corr_matrix, max_rows=None)
            
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

            # Data Analysis Questions
            st.subheader("Ask Questions About Your Data")
            if st.session_state.data is not None:
                question = st.text_input("What would you like to know about your data?", 
                                       placeholder="e.g., What are the main trends in this dataset?")
                
                if st.button("Get Insights"):
                    if question:
                        with st.spinner("Analyzing your data..."):
                            insights = generate_data_insights(question, st.session_state.data)
                            st.write("Analysis:")
                            st.write(insights)
                    else:
                        st.warning("Please enter a question about your data.")

            # Continue Conversation Section
            st.subheader("🤖 Continue the Conversation")
            st.write("Ask follow-up questions or explore different aspects of your data analysis.")
            
            # Initialize chat history in session state if it doesn't exist
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a follow-up question or request additional analysis..."):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Generate AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = generate_data_insights(prompt, st.session_state.data)
                        st.write(response)
                        # Add AI response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Add a clear chat button
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a CSV or Excel file to get started.") 