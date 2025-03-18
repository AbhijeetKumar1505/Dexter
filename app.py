import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import StringIO, BytesIO
import base64
from sklearn.impute import SimpleImputer
import os
import re
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Data Cleaning Tool",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #4B5D67;
    }
    .stButton button {
        background-color: #4B9CD3;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #357AB7;
    }
    .st-emotion-cache-16txtl3 {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'cleaning_history' not in st.session_state:
    st.session_state.cleaning_history = []
if 'file_name' not in st.session_state:
    st.session_state.file_name = None

# Functions for data cleaning operations
def remove_duplicates(df):
    """Remove duplicate rows from the dataframe"""
    rows_before = len(df)
    df = df.drop_duplicates()
    rows_after = len(df)
    return df, f"Removed {rows_before - rows_after} duplicate rows"

def fill_missing_values(df, method, columns):
    """Fill missing values using the specified method"""
    for col in columns:
        if method == "Mean":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
        elif method == "Median":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
        elif method == "Mode":
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
        elif method == "Zero":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
        elif method == "Forward Fill":
            df[col] = df[col].ffill()
        elif method == "Backward Fill":
            df[col] = df[col].bfill()
        elif method == "Custom Value":
            df[col] = df[col].fillna(st.session_state.custom_value)
    
    return df, f"Filled missing values in {', '.join(columns)} using {method}"

def remove_outliers(df, columns, method, threshold=1.5):
    """Remove outliers from the specified columns"""
    rows_before = len(df)
    
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        if method == "IQR":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == "Z-Score":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores <= threshold]
            
    rows_after = len(df)
    return df, f"Removed {rows_before - rows_after} outliers from {', '.join(columns)} using {method}"

def standardize_column(df, columns):
    """Standardize columns (z-score normalization)"""
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
    
    return df, f"Standardized columns: {', '.join(columns)}"

def normalize_column(df, columns):
    """Normalize columns to range [0, 1]"""
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df, f"Normalized columns: {', '.join(columns)}"

def rename_columns(df, rename_dict):
    """Rename columns based on the provided dictionary"""
    df = df.rename(columns=rename_dict)
    return df, f"Renamed columns: {', '.join([f'{old} → {new}' for old, new in rename_dict.items()])}"

def create_download_link(df, filename, text):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="display: inline-block; padding: 0.5em 1em; color: white; background-color: #4CAF50; text-decoration: none; border-radius: 4px;">{text}</a>'
    return href

def ensure_consistent_types(df):
    """Ensure all object-type columns are converted to strings"""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    return df

def main():
    st.title("🧹 Data Cleaning Tool")
    
    # Sidebar
    st.sidebar.header("Operations")
    
    # File upload section
    st.sidebar.subheader("1. Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None and (st.session_state.file_name != uploaded_file.name or st.session_state.data is None):
        st.session_state.file_name = uploaded_file.name
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
                
            st.session_state.data = data
            st.session_state.original_data = data.copy()
            st.session_state.cleaning_history = []
            st.sidebar.success(f"Successfully loaded {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # If data is loaded, show data statistics and cleaning options
    if st.session_state.data is not None:
        data = ensure_consistent_types(st.session_state.data)
        
        # Data overview
        st.header("📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", data.shape[0])
        col2.metric("Columns", data.shape[1])
        col3.metric("Missing Values", data.isna().sum().sum())
        col4.metric("Duplicate Rows", data.duplicated().sum())
        
        # Preview data
        with st.expander("Preview Data", expanded=True):
            st.dataframe(data.head(10), use_container_width=True)
            
        # Column information
        with st.expander("Column Information"):
            col_stats = pd.DataFrame({
                'Type': data.dtypes,
                'Unique Values': data.nunique(),
                'Missing Values': data.isna().sum(),
                'Missing (%)': (data.isna().sum() / len(data) * 100).round(2)
            })
            st.dataframe(col_stats, use_container_width=True)
        
        # Data visualization
        with st.expander("Data Visualization"):
            viz_col1, viz_col2 = st.columns(2)
            
            # Select column to visualize
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = data.select_dtypes(exclude=['number']).columns.tolist()
            
            if numeric_cols:
                with viz_col1:
                    st.subheader("Numeric Column Distribution")
                    sel_num_col = st.selectbox("Select numeric column", numeric_cols)
                    fig = px.histogram(data, x=sel_num_col, nbins=20)
                    st.plotly_chart(fig, use_container_width=True)
            
            if categorical_cols:
                with viz_col2:
                    st.subheader("Categorical Column Distribution")
                    sel_cat_col = st.selectbox("Select categorical column", categorical_cols)
                    value_counts = data[sel_cat_col].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, labels={'x': sel_cat_col, 'y': 'Count'})
                    st.plotly_chart(fig, use_container_width=True)
        
        # Cleaning operations
        st.sidebar.subheader("2. Select Cleaning Operations")
        
        cleaning_option = st.sidebar.selectbox(
            "Choose operation", 
            ["Remove Duplicates", 
             "Handle Missing Values", 
             "Remove Outliers", 
             "Standardize Data", 
             "Normalize Data", 
             "Rename Columns", 
             "Drop Columns"]
        )
        
        # Handle different cleaning operations
        if cleaning_option == "Remove Duplicates":
            st.sidebar.markdown("Remove duplicate rows from the data")
            if st.sidebar.button("Remove Duplicates"):
                data, message = remove_duplicates(data)
                st.session_state.data = data
                st.session_state.cleaning_history.append(message)
                st.experimental_rerun()
        
        elif cleaning_option == "Handle Missing Values":
            cols_with_missing = data.columns[data.isna().any()].tolist()
            
            if cols_with_missing:
                st.sidebar.markdown("Columns with missing values:")
                selected_cols = st.sidebar.multiselect("Select columns", cols_with_missing, default=cols_with_missing)
                
                fill_method = st.sidebar.selectbox(
                    "Fill method", 
                    ["Mean", "Median", "Mode", "Zero", "Forward Fill", "Backward Fill", "Custom Value"]
                )
                
                if fill_method == "Custom Value":
                    st.session_state.custom_value = st.sidebar.text_input("Enter custom value")
                
                if st.sidebar.button("Fill Missing Values") and selected_cols:
                    data, message = fill_missing_values(data, fill_method, selected_cols)
                    st.session_state.data = data
                    st.session_state.cleaning_history.append(message)
                    st.experimental_rerun()
            else:
                st.sidebar.info("No missing values found in the data")
        
        elif cleaning_option == "Remove Outliers":
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                selected_cols = st.sidebar.multiselect("Select columns", numeric_cols)
                outlier_method = st.sidebar.selectbox("Method", ["IQR", "Z-Score"])
                
                if outlier_method == "IQR":
                    threshold = st.sidebar.slider("IQR Threshold", 1.0, 3.0, 1.5, 0.1)
                else:
                    threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)
                
                if st.sidebar.button("Remove Outliers") and selected_cols:
                    data, message = remove_outliers(data, selected_cols, outlier_method, threshold)
                    st.session_state.data = data
                    st.session_state.cleaning_history.append(message)
                    st.experimental_rerun()
            else:
                st.sidebar.info("No numeric columns found in the data")
        
        elif cleaning_option == "Standardize Data":
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                selected_cols = st.sidebar.multiselect("Select columns to standardize", numeric_cols)
                
                if st.sidebar.button("Standardize") and selected_cols:
                    data, message = standardize_column(data, selected_cols)
                    st.session_state.data = data
                    st.session_state.cleaning_history.append(message)
                    st.experimental_rerun()
            else:
                st.sidebar.info("No numeric columns found in the data")
        
        elif cleaning_option == "Normalize Data":
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                selected_cols = st.sidebar.multiselect("Select columns to normalize", numeric_cols)
                
                if st.sidebar.button("Normalize") and selected_cols:
                    data, message = normalize_column(data, selected_cols)
                    st.session_state.data = data
                    st.session_state.cleaning_history.append(message)
                    st.experimental_rerun()
            else:
                st.sidebar.info("No numeric columns found in the data")
        
        elif cleaning_option == "Rename Columns":
            col_rename = {}
            for col in data.columns:
                new_name = st.sidebar.text_input(f"Rename '{col}'", col)
                if new_name != col:
                    col_rename[col] = new_name
            
            if st.sidebar.button("Rename Columns") and col_rename:
                data, message = rename_columns(data, col_rename)
                st.session_state.data = data
                st.session_state.cleaning_history.append(message)
                st.experimental_rerun()
        
        elif cleaning_option == "Drop Columns":
            selected_cols = st.sidebar.multiselect("Select columns to drop", data.columns)
            
            if st.sidebar.button("Drop Columns") and selected_cols:
                rows_before = len(data)
                data = data.drop(columns=selected_cols)
                rows_after = len(data)
                
                message = f"Dropped columns: {', '.join(selected_cols)}"
                st.session_state.data = data
                st.session_state.cleaning_history.append(message)
                st.experimental_rerun()
        
        # History of cleaning operations
        if st.session_state.cleaning_history:
            with st.expander("Cleaning History", expanded=True):
                for i, action in enumerate(st.session_state.cleaning_history):
                    st.write(f"{i+1}. {action}")
        
        # Reset and download options
        st.sidebar.subheader("3. Final Actions")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("Reset Data"):
                st.session_state.data = st.session_state.original_data.copy()
                st.session_state.cleaning_history = []
                st.experimental_rerun()
        
        with col2:
            filename = f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            download_button_html = create_download_link(data, filename, "Download Data")
            st.markdown(download_button_html, unsafe_allow_html=True)
    
    else:
        # Instructions for first-time users
        st.markdown("""
        ## Welcome to the Data Cleaning Tool
        
        This tool helps you clean and prepare your data for analysis. Follow these steps:
        
        1. **Upload Data**: Use the sidebar to upload a CSV or Excel file
        2. **Explore Data**: View statistics and visualizations of your data
        3. **Clean Data**: Apply various cleaning operations to improve data quality
        4. **Download Results**: Get your cleaned data for further analysis
        
        ### Features:
        
        - Remove duplicate rows
        - Handle missing values
        - Remove outliers
        - Standardize or normalize numeric columns
        - Rename or drop columns
        - Track cleaning history
        
        ### Supported File Formats:
        
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        """)

if __name__ == "__main__":
    main() 