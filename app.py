import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import chromadb
import requests
import os
import re
import io
import time
import numpy as np
import json
import scipy.stats as stats
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv

# Import data preprocessing module
from data_preprocessing import (
    DataCleaner, 
    TextPreprocessor, 
    FeatureEngineer, 
    create_preprocessing_pipeline,
    get_feature_names
)

# Import ML enhancements
from ml_ui import MLUI
from ml_enhancements import (
    AutomatedFeatureEngineering, ModelComparator, ModelInterpreter,
    UnsupervisedLearning, get_available_models, create_preprocessing_pipeline,
    hyperparameter_tuning
)

# Load environment variables from .env file
load_dotenv()

# Model constants
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
KIMI_MODEL = "Moonshot/Kimi-K2-Instruct"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

def process_with_mistral(prompt: str, analysis: str, api_key: str = None) -> Dict[str, Any]:
    """
    Process Kimi's analysis through Mistral for chat completion.
    
    Args:
        prompt: Original user prompt
        analysis: Analysis from Kimi
        api_key: Together API key
        
    Returns:
        Dictionary with final response
    """
    api_key = api_key or TOGETHER_API_KEY
    if not api_key:
        return {"error": "API key not provided"}
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_prompt = """You are a helpful assistant that refines and presents data analysis in a clear, 
    conversational way. Make the analysis more engaging and easier to understand for non-technical users."""
    
    user_prompt = f"""Original user query: {prompt}
    
    Data analysis from Kimi:
    {analysis}
    
    Please refine this analysis to be more conversational and user-friendly while maintaining all key insights."""
    
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1500,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1
    }
    
    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return {
                "response": result["choices"][0]["message"]["content"],
                "model": MISTRAL_MODEL,
                "usage": result.get("usage", {})
            }
        else:
            return {"error": "Unexpected response format from Mistral API"}
    except Exception as e:
        return {"error": f"Error querying Mistral API: {str(e)}"}


def analyze_data(prompt: str, api_key: str = None, df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Pipeline that first analyzes data with Kimi and then refines the response with Mistral.
    
    Args:
        prompt: User's data analysis request
        api_key: Together API key
        df: Optional DataFrame to analyze
        
    Returns:
        Dictionary with the final response and metadata
    """
    # Step 1: Get analysis from Kimi
    st.info("🔍 Analyzing data  ...")
    kimi_result = analyze_with_kimi(prompt, api_key, df)
    
    if "error" in kimi_result:
        return {"error": f"Data analysis failed: {kimi_result['error']}"}
    
    # Step 2: Process with Mistral
    st.info("💬 Refining response with Mistral...")
    mistral_result = process_with_mistral(
        prompt=prompt,
        analysis=kimi_result["analysis"],
        api_key=api_key
    )
    
    if "error" in mistral_result:
        # If Mistral fails, return Kimi's analysis as fallback
        return {
            "response": kimi_result["analysis"],
            "analysis": kimi_result["analysis"],
            "models": [kimi_result["model"]],
            "usage": {"kimi": kimi_result.get("usage", {})}
        }
    
    # Return combined results
    return {
        "response": mistral_result["response"],
        "analysis": kimi_result["analysis"],
        "models": [kimi_result["model"], mistral_result["model"]],
        "usage": {
            "kimi": kimi_result.get("usage", {}),
            "mistral": mistral_result.get("usage", {})
        }
    }


def analyze_with_kimi(prompt: str, api_key: str = None, df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Send a prompt to Kimi for data analysis.
    
    Args:
        prompt: The analysis prompt
        api_key: Together API key
        df: Optional DataFrame to include in the context
        
    Returns:
        Dictionary with analysis results
    """
    api_key = api_key or TOGETHER_API_KEY
    if not api_key:
        return {"error": "API key not provided"}
    
    # Prepare the system message with data context if available
    system_message = "You are a data analysis expert. Analyze the following data and provide insights:"
    
    if df is not None:
        data_summary = f"""
        Data Summary:
        - Shape: {df.shape}
        - Columns: {', '.join(df.columns)}
        - Sample Data (first 3 rows):\n{df.head(3).to_string()}
        - Description (statistics):\n{df.describe().to_string()}
        """
        system_message += data_summary
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": KIMI_MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1
    }
    
    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return {
                "analysis": result["choices"][0]["message"]["content"],
                "model": KIMI_MODEL,
                "usage": result.get("usage", {})
            }
        else:
            return {"error": "Unexpected response format from Kimi API"}
    except Exception as e:
        return {"error": f"Error querying Kimi API: {str(e)}"}

# Debug: Check if API key is loaded
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    st.error("TOGETHER_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()
else:
    st.sidebar.success("API key loaded successfully")

# Together AI Embedding Function
class TogetherAIEmbeddingFunction:
    def __init__(self, api_key=None, model_name="BAAI/bge-base-en-v1.5"):
        """
        Initialize the embedding function with a compatible model.
        Supported models: 'BAAI/bge-base-en-v1.5', 'sentence-transformers/all-mpnet-base-v2', 'BAAI/bge-large-en-v1.5'
        """
        self.api_key = os.getenv("TOGETHER_API_KEY")
        self.model_name = model_name
        self.api_url = "https://api.together.xyz/v1/embeddings"
        
    def name(self):
        return "together-ai"  # Required by ChromaDB
        
    def __call__(self, input):
        if not isinstance(input, list):
            input = [input]
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        embeddings = []
        for text in input:
            data = {
                "model": self.model_name,
                "input": text
            }
            
            # Debug: Print the request details
            print(f"Sending request to Together API with model: {self.model_name}")
            print(f"API Key (first 10 chars): {self.api_key[:10]}...")
            print(f"Input text: {text[:100]}...")
            
            try:
                response = requests.post(self.api_url, json=data, headers=headers, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                # Debug: Print the response status and keys
                print(f"Response status: {response.status_code}")
                print(f"Response keys: {result.keys()}")
                
                embedding = result["data"][0]["embedding"]
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"Error details: {str(e)}")
                print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
                raise ValueError(f"Error from Together API: {response.text}")
            
        return np.array(embeddings).tolist()

st.set_page_config(page_title="AI Data Assistant", layout="wide")
st.title("AI Data Assistant")

# Session state for data and chat
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# File upload and processing
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Store in session state
        st.session_state.df = df
        st.session_state.file_uploaded = True
        st.success(f"Successfully loaded {uploaded_file.name} with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Data editor for making changes
        st.subheader("Data Editor")
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            key="data_editor"
        )
        
        # Save button for edited data
        if st.button("Save Changes"):
            st.session_state.df = edited_df
            st.success("Changes saved to session!")
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.session_state.file_uploaded = False
else:
    st.info("Please upload a CSV or Excel file to get started.")

# TogetherAI API integration
# Load API key from environment variables
TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY')
if not TOGETHER_API_KEY:
    st.error("TOGETHER_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
# Using Mistral 7B for chat completions
TOGETHER_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # Using Mistral 7B for chat

def query_agent(prompt: str, api_key: str = None, model: str = None, is_visualization: bool = False, df: pd.DataFrame = None) -> str:
    """
    Query the AI agent with a prompt and optional DataFrame for context.
    
    Args:
        prompt: The user's query or prompt
        api_key: Together AI API key (optional, falls back to TOGETHER_API_KEY)
        model: Model to use (optional, falls back to TOGETHER_MODEL)
        is_visualization: Whether this is a visualization request
        df: Optional DataFrame to include in the context
        
    Returns:
        str: The AI's response
    """
    api_key = api_key or TOGETHER_API_KEY
    model = model or TOGETHER_MODEL
    
    if not api_key:
        return "[Error: TogetherAI API key not set in environment variable TOGETHER_API_KEY.]"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Enhanced system prompt based on context
    system_prompt = (
        "You are a helpful data science assistant. "
        "For data analysis tasks, provide clear, concise Python code using pandas/plotly. "
        "For visualizations, suggest appropriate chart types and generate complete, runnable code. "
        "Always include proper error handling and data validation.\n\n"
    )
    
    # Add DataFrame information to the prompt if provided
    if df is not None:
        data_info = f"""
        Current DataFrame Information:
        - Shape: {df.shape}
        - Columns: {', '.join(df.columns)}
        - Sample data (first 3 rows):
        {df.head(3).to_string()}
        - Column data types:
        {df.dtypes}
        - Basic statistics:
        {df.describe(include='all').to_string()}
        """
        system_prompt += data_info
    
    if is_visualization:
        system_prompt += (
            "\nFor visualization requests, suggest the most appropriate chart type and "
            "provide complete, runnable Plotly code. Consider the data type and distribution "
            "when choosing visualizations. If a specific column is mentioned, focus on that column. "
            "For categorical data, consider bar charts or pie charts. For numerical data, consider "
            "histograms, box plots, or scatter plots for relationships between variables."
        )
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,  # Increased for more detailed responses
        "temperature": 0.3,  # Slightly higher for more creative suggestions
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "stop": ["```"]  # Stop at code blocks to prevent run-on responses
    }
    
    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return f"[Error: Unexpected response format from Together AI: {result}]"
            
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error: {e}"
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            error_msg += f"\nResponse: {e.response.text}"
        return f"[LLM Error: {error_msg}]"
    except Exception as e:
        return f"[LLM Error: {str(e)}]"

# Prompt templates
PROMPT_TEMPLATES = {
    "cleaning": "Remove nulls from column {col} and standardize names.",
    "editing": "Drop column {col} and rename {old} to {new}.",
    "grouping": "Group by {group_col} and find average of {agg_col}.",
    "visualization": "Plot {col} over time grouped by {group_col}."
}

# Initialize embedding function with a compatible model
embedding_func = TogetherAIEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")

# Chroma vector memory setup
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()

# Delete existing collection if it exists and create a new one with the correct embedding function
def initialize_memory_collection():
    """Initialize the memory collection for chat history."""
    try:
        # Try to delete the existing collection if it exists
        try:
            st.session_state.chroma_client.delete_collection("chat_memory")
        except Exception as e:
            # Collection didn't exist, which is fine
            pass
        
        # Create a new collection with our embedding function
        st.session_state.memory_collection = st.session_state.chroma_client.create_collection(
            "chat_memory",
            embedding_function=embedding_func
        )
        return True
    except Exception as e:
        st.warning(f"Could not initialize memory collection: {str(e)}")
        return False

# Initialize memory collection
if 'memory_collection' not in st.session_state:
    initialize_memory_collection()

# Helper: Add to memory
def add_to_memory(query: str, response: str) -> None:
    """
    Add a query-response pair to the memory collection.
    
    Args:
        query: The user's query
        response: The assistant's response
    """
    try:
        # Check if memory collection exists
        if 'memory_collection' not in st.session_state:
            # Try to initialize the collection if it doesn't exist
            try:
                embedding_func = TogetherAIEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")
                st.session_state.memory_collection = st.session_state.chroma_client.create_collection(
                    "chat_memory",
                    embedding_function=embedding_func
                )
            except Exception as e:
                st.warning(f"Could not initialize memory collection: {str(e)}")
                return
        
        # Prepare the document to store
        doc_id = f"{len(st.session_state.chat_history)}_{int(time.time())}"
        document = f"Q: {query}\nA: {response}"
        
        # Try to add to memory
        st.session_state.memory_collection.add(
            documents=[document],
            ids=[doc_id]
        )
        
    except Exception as e:
        # If adding to memory fails, just log the error and continue
        st.warning(f"Could not add to memory: {str(e)}")
        # Fallback: store in chat history only
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("assistant", response))

# Helper: Retrieve context
def retrieve_context(query, k=3):
    """
    Retrieve relevant context from memory using the embedding function.
    
    Args:
        query: The query to search for
        k: Number of results to return
        
    Returns:
        String containing the concatenated context
    """
    try:
        if 'memory_collection' not in st.session_state:
            return ""
            
        results = st.session_state.memory_collection.query(
            query_texts=[query],
            n_results=min(k, 5)  # Limit to max 5 results
        )
        
        # Extract documents from results
        documents = results.get('documents', [[]])
        if not documents or not documents[0]:
            return ""
            
        # Join documents with newlines
        return '\n'.join(doc for doc in documents[0] if doc)
        
    except Exception as e:
        st.warning(f"Could not retrieve context: {str(e)}")
        return ""

def extract_code_blocks(text):
    # Extract Python code blocks from markdown
    code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
    return [cb.strip() for cb in code_blocks]

def execute_code_on_df(code, df):
    # Safe execution: restrict builtins, only allow df
    local_vars = {'df': df.copy()}
    try:
        exec(code, {'__builtins__': {}}, local_vars)
        result = local_vars.get('df', None)
        return result, None
    except Exception as e:
        return None, str(e)

def data_preprocessing_ui(df):
    """User interface for data preprocessing"""
    st.header("Data Preprocessing")
    
    # Display current data info
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    # Data cleaning options
    with st.expander("🛠️ Data Cleaning", expanded=True):
        st.subheader("Data Cleaning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            drop_duplicates = st.checkbox("Remove duplicate rows", value=True)
            handle_missing = st.selectbox(
                "Handle missing values",
                ["drop", "mean", "median", "mode", "knn", "ffill", "bfill"],
                index=2
            )
            
        with col2:
            outlier_method = st.selectbox(
                "Handle outliers",
                ["None", "zscore", "iqr"],
                index=0
            )
            
        # Identify datetime columns
        datetime_cols = st.multiselect(
            "Select datetime columns",
            df.select_dtypes(include=['datetime', 'object']).columns.tolist(),
            help="Select columns containing datetime data"
        )
    
    # Text preprocessing
    with st.expander("📝 Text Processing", expanded=False):
        st.subheader("Text Processing")
        
        text_cols = st.multiselect(
            "Select text columns",
            df.select_dtypes(include=['object']).columns.tolist(),
            help="Select columns containing text data"
        )
        
        if text_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                remove_punct = st.checkbox("Remove punctuation", value=True)
                to_lower = st.checkbox("Convert to lowercase", value=True)
                remove_stopwords = st.checkbox("Remove stopwords", value=True)
                
            with col2:
                lemmatize = st.checkbox("Lemmatize words", value=True)
                remove_numbers = st.checkbox("Remove numbers", value=True)
                remove_special = st.checkbox("Remove special characters", value=True)
    
    # Feature engineering
    with st.expander("⚙️ Feature Engineering", expanded=False):
        st.subheader("Feature Engineering")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        # Convert to sets for difference operation
        all_cat_cols = set(df.select_dtypes(include=['object', 'category']).columns)
        text_cols_set = set(text_cols) if text_cols else set()
        cat_cols = list(all_cat_cols - text_cols_set)
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_interactions = st.checkbox("Create interaction features", value=True)
            create_polynomials = st.checkbox("Create polynomial features", value=True)
            
        with col2:
            poly_degree = st.number_input(
                "Polynomial degree",
                min_value=2,
                max_value=5,
                value=2,
                step=1
            )
            create_stats = st.checkbox("Create statistical features", value=True)
    
    # Apply preprocessing
    if st.button("Apply Preprocessing"):
        with st.spinner("Processing data..."):
            try:
                # Initialize cleaner
                cleaner = DataCleaner(
                    drop_duplicates=drop_duplicates,
                    handle_missing=handle_missing if handle_missing != "None" else None,
                    outlier_method=outlier_method if outlier_method != "None" else None,
                    datetime_columns=datetime_cols
                )
                
                # Apply data cleaning
                df_cleaned = cleaner.fit_transform(df.copy())
                
                # Apply text preprocessing if text columns selected
                if text_cols:
                    text_preprocessor = TextPreprocessor(
                        text_columns=text_cols,
                        remove_punctuation=remove_punct,
                        to_lowercase=to_lower,
                        remove_stopwords=remove_stopwords,
                        lemmatize=lemmatize,
                        remove_numbers=remove_numbers,
                        remove_special_chars=remove_special
                    )
                    df_cleaned = text_preprocessor.fit_transform(df_cleaned)
                
                # Apply feature engineering
                feature_engineer = FeatureEngineer(
                    create_interactions=create_interactions,
                    polynomial_degree=poly_degree,
                    create_statistical_features=create_stats,
                    numeric_columns=numeric_cols,
                    categorical_columns=cat_cols
                )
                
                # Store the processed data in session state
                st.session_state.processed_df = df_cleaned
                st.session_state.feature_engineer = feature_engineer
                
                st.success("Data preprocessing completed successfully!")
                
                # Show processed data preview
                st.subheader("Processed Data Preview")
                st.dataframe(df_cleaned.head())
                
                # Show data summary
                st.subheader("Data Summary")
                st.json({
                    "Original shape": str(df.shape),
                    "Processed shape": str(df_cleaned.shape),
                    "Missing values": int(df_cleaned.isnull().sum().sum()),
                    "Numeric columns": len(df_cleaned.select_dtypes(include=['number']).columns),
                    "Categorical columns": len(df_cleaned.select_dtypes(include=['object', 'category']).columns)
                })
                
                # Download button for processed data
                csv = df_cleaned.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download processed data",
                    data=csv,
                    file_name='processed_data.csv',
                    mime='text/csv'
                )
                
            except Exception as e:
                st.error(f"Error during preprocessing: {str(e)}")
                st.exception(e)

# Import the code generation module
from code_generation import CodeGenerator

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data Analysis", 
    "Data Preprocessing", 
    "Visualization", 
    "Machine Learning", 
    "Chat",
    "Code Generation"
])

with tab1:
    # Data Analysis tab
    st.header("Data Analysis")
    if 'df' in st.session_state and st.session_state.df is not None:
        st.dataframe(st.session_state.df.head())
        
        # Basic statistics
        st.subheader("Basic Statistics")
        st.write(st.session_state.df.describe())
        
        # Data types info
        st.subheader("Data Types")
        st.write(st.session_state.df.dtypes)
    else:
        st.warning("Please upload a dataset to begin analysis.")

with tab2:
    # Data Preprocessing tab
    if 'df' in st.session_state and st.session_state.df is not None:
        data_preprocessing_ui(st.session_state.df)
    else:
        st.warning("Please upload a dataset to begin preprocessing.")

with tab3:
    # Enhanced ML Interface
    st.header("Advanced Machine Learning")
    
    if 'df' in st.session_state and st.session_state.df is not None:
        # Initialize ML UI with the current dataframe
        ml_ui = MLUI(st.session_state.df)
        
        # Task selection
        target_col, feature_cols, test_size, random_state = ml_ui.select_task()
        
        # Feature engineering
        feature_eng_params = ml_ui.feature_engineering_ui()
        
    else:
        st.warning("Please upload a dataset to create visualizations.")

with tab4:
    # Machine Learning tab
    st.header("Machine Learning")
    if 'df' in st.session_state and st.session_state.df is not None:
        # Initialize MLUI if not already done
        if 'ml_ui' not in st.session_state:
            st.session_state.ml_ui = MLUI(st.session_state.df)
        
        # Show ML interface
        st.session_state.ml_ui.select_task()
        
        # Get feature engineering parameters
        feature_eng_params = st.session_state.ml_ui.feature_engineering_ui()
        
        # Get model selection and training parameters
        param_grids = st.session_state.ml_ui.model_selection_ui()
        
        # Add a button to train models
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                st.session_state.ml_ui.train_and_compare_models(param_grids)
        
        # Show model interpretation if models are trained
        if hasattr(st.session_state.ml_ui, 'best_model'):
            st.session_state.ml_ui.model_interpretation_ui()
            
            # Add a section for unsupervised learning
            st.markdown("---")
            st.session_state.ml_ui.unsupervised_learning_ui()
    else:
        st.warning("Please upload a dataset to use machine learning features.")

with tab5:
    # Chat interface
    st.header("Chat with AI Assistant")
    
    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI data assistant. You can ask me to analyze your data, create visualizations, or help with machine learning tasks."}
        ]
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your data"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Get AI response using the query_agent function
                if 'df' in st.session_state and st.session_state.df is not None:
                    try:
                        # Get context from memory
                        context = retrieve_context(prompt)
                        
                        # Enhance the prompt with context
                        enhanced_prompt = f"Context from previous conversation:\n{context}\n\nUser's current question: {prompt}"
                        
                        # Get response from the AI model
                        ai_response = query_agent(
                            prompt=enhanced_prompt,
                            df=st.session_state.df
                        )
                        
                        # Validate the response
                        if not ai_response or ai_response.strip() == '':
                            ai_response = "I'm sorry, I didn't receive a valid response. Please try rephrasing your question."
                        elif ai_response.startswith('[Error:') or ai_response.startswith('[LLM Error:'):
                            st.error(f"Error from AI service: {ai_response}")
                            ai_response = "I encountered an error while processing your request. Please check the error message above and try again."
                        
                        # Add to memory if we have a valid response
                        if not ai_response.startswith('I\'m sorry') and not ai_response.startswith('I encountered'):
                            add_to_memory(prompt, ai_response)
                            
                    except Exception as e:
                        st.error(f"Error processing your request: {str(e)}")
                        ai_response = "I'm sorry, I encountered an error while processing your request. Please try again."
                else:
                    ai_response = "Please upload a dataset first so I can help you analyze it. You can use the 'Upload Data' tab to get started."
                
                # Process the response to handle markdown and code blocks
                full_response = ai_response
                
            except Exception as e:
                st.error(f"An error occurred while generating the response: {str(e)}")
                full_response = "I'm sorry, I encountered an error while processing your request. Please try again."
            
            # Display the response
            response_placeholder.markdown(full_response)
            
            # Extract and display any code blocks in the response
            code_blocks = extract_code_blocks(full_response)
            if code_blocks and 'df' in st.session_state and st.session_state.df is not None:
                for i, code in enumerate(code_blocks):
                    with st.expander(f"View and Run Code #{i+1}"):
                        st.code(code, language="python")
                        
                        if st.button(f"Run Code #{i+1}", key=f"run_code_{i}"):
                            try:
                                # Execute the code in a safe environment
                                local_vars = {'df': st.session_state.df.copy()}
                                exec(code, globals(), local_vars)
                                
                                # If the code modifies the dataframe, update the session state
                                if 'df' in local_vars and local_vars['df'] is not None:
                                    st.session_state.df = local_vars['df']
                                    st.success("Code executed successfully! The DataFrame has been updated.")
                                    
                                    # Rerun to update the UI
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error executing code: {str(e)}")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Display chat history and code execution
for idx, (role, msg) in enumerate(st.session_state.chat_history):
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Assistant:** {msg}")
        # If LLM response, check for code blocks
        code_blocks = extract_code_blocks(msg)
        if code_blocks:
            for code in code_blocks:
                with st.expander("View and Run Code"):
                    st.code(code, language="python")
                    if st.button("Run Code", key=f"run_{idx}_{hash(code)}"):
                        try:
                            # Execute the code in a safe environment
                            local_vars = {'df': st.session_state.df.copy() if 'df' in st.session_state else None}
                            exec(code, globals(), local_vars)
                            
                            # If the code modifies the dataframe, update the session state
                            if 'df' in local_vars and local_vars['df'] is not None:
                                st.session_state.df = local_vars['df']
                                st.success("DataFrame updated successfully!")
                        except Exception as e:
                            st.error(f"Error executing code: {str(e)}")
                            st.exception(e)
        code_blocks = extract_code_blocks(msg)
        if code_blocks and st.session_state.df is not None:
            for i, code in enumerate(code_blocks):
                with st.expander(f"Show suggested code #{i+1}"):
                    st.code(code, language="python")
                    if st.button(f"Run code #{i+1} on data", key=f"run_{idx}_{i}"):
                        result, error = execute_code_on_df(code, st.session_state.df)
                        if error:
                            st.error(f"Execution error: {error}")
                        elif result is not None:
                            st.session_state.df = result
                            st.success("Code executed and DataFrame updated.")
                            st.dataframe(result.head(100)) 

# Enhanced Visualization Engine with Hybrid Models
if st.session_state.df is not None:
    st.subheader("📊 Advanced Data Visualization")
    
    # Get column types for better visualization handling
    numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = st.session_state.df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Ensure we have column type lists even if empty
    numeric_cols = numeric_cols if numeric_cols else []
    categorical_cols = categorical_cols if categorical_cols else []
    datetime_cols = datetime_cols if datetime_cols else []
    
    # Basic data info
    st.sidebar.subheader("Visualization Options")
    
    # Add a selector for visualization type with more chart options
    viz_type = st.sidebar.selectbox(
        "Select Visualization Type",
        [
            "Auto-detect", 
            "Scatter Plot", 
            "Line Chart", 
            "Bar Chart", 
            "Histogram", 
            "Box Plot", 
            "Violin Plot", 
            "Heatmap", 
            "Correlation Matrix", 
            "Treemap",
            "Sunburst",
            "Parallel Coordinates",
            "Parallel Categories",
            "Density Contour",
            "Geographic Map",
            "Dashboard View"
        ]
    )
    
    # Add interactive controls
    use_interactivity = st.sidebar.checkbox("Enable Interactive Features", value=True)
    
    # Add color and styling options
    color_theme = st.sidebar.selectbox(
        "Color Theme",
        ["Plotly", "Seaborn", "Plotly Dark", "Plotly White", "Plotly Vivid", "D3", "GGPlot2"]
    )
    
    # Add chart customization options
    st.sidebar.subheader("Chart Customization")
    chart_width = st.sidebar.slider("Chart Width", 400, 1200, 800, 50)
    chart_height = st.sidebar.slider("Chart Height", 300, 1000, 500, 50)
    show_legend = st.sidebar.checkbox("Show Legend", value=True)
    show_grid = st.sidebar.checkbox("Show Grid", value=True)
    
    # Animation settings
    if use_interactivity:
        st.sidebar.subheader("Animation Settings")
        animation_speed = st.sidebar.slider("Animation Speed", 100, 2000, 500, 100)
        transition_duration = st.sidebar.slider("Transition Duration (ms)", 100, 1000, 300, 50)
    
    # Set the color theme
    if color_theme == "Seaborn":
        px.defaults.template = "seaborn"
    elif color_theme == "Plotly Dark":
        px.defaults.template = "plotly_dark"
    elif color_theme == "Plotly White":
        px.defaults.template = "plotly_white"
    elif color_theme == "Plotly Vivid":
        px.defaults.template = "plotly"
        px.defaults.color_discrete_sequence = px.colors.qualitative.Vivid
    else:
        px.defaults.template = "plotly"
    # Basic data info in sidebar
    st.sidebar.write(f"Rows: {len(st.session_state.df)}")
    st.sidebar.write(f"Columns: {', '.join(st.session_state.df.columns.tolist())}")
    
    # Initialize visualization controls with default values
    x_axis = ""
    y_axis = ""
    color_by = "None"
    facet_col = "None"
    hover_data = []
    animation_frame = None
    
    # Get all column names for general use
    all_columns = st.session_state.df.columns.tolist()
    
    # Main visualization controls
    st.subheader("Visualization Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        # X-axis selection - allow any column
        x_axis = st.selectbox("X-axis", [""] + all_columns)
        
        # Y-axis selection - prefer numeric columns but allow any if none available
        y_options = [""]  # Start with empty option
        if numeric_cols:
            y_options.extend(numeric_cols)
        else:
            y_options.extend(all_columns)
        y_axis = st.selectbox("Y-axis", y_options)
        
        # Color by selection - prefer categorical columns but allow any if none available
        color_options = ["None"]  # Start with None option
        if categorical_cols:
            color_options.extend(categorical_cols)
        else:
            color_options.extend(all_columns)
        color_by = st.selectbox("Color by", color_options)
    
    with col2:
        # Facet column selection
        facet_options = ["None"]
        if categorical_cols:
            facet_options.extend(categorical_cols)
        facet_col = st.selectbox("Facet Column", facet_options) if categorical_cols else "None"
        
        # Hover data selection
        hover_data = st.multiselect("Additional Hover Data", 
                                  all_columns,
                                  default=[])
        
        # Animation frame selection (only show if datetime columns exist)
        if datetime_cols:
            animation_options = ["None"] + datetime_cols
            animation_frame = st.selectbox("Animate by", animation_options, key="animation_frame")
            if animation_frame == "None":
                animation_frame = None
        else:
            animation_frame = None
    
    # Set default values if not selected
    if color_by == "None":
        color_by = None
    if facet_col == "None":
        facet_col = None
    
    # Initialize figure variable
    fig = None
    
    # Function to apply common figure settings
    def apply_figure_settings(fig, title=None, x_title=None, y_title=None):
        if title:
            fig.update_layout(title=title)
        if x_title:
            fig.update_xaxes(title=x_title)
        if y_title:
            fig.update_yaxes(title=y_title)
            
        fig.update_layout(
            width=chart_width,
            height=chart_height,
            showlegend=show_legend,
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=show_grid, gridcolor='lightgray'),
            yaxis=dict(showgrid=show_grid, gridcolor='lightgray'),
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode='closest'
        )
        
        if use_interactivity and hasattr(fig, 'layout') and hasattr(fig.layout, 'updatemenus'):
            fig.update_layout(
                updatemenus=[dict(
                    type="buttons",
                    direction="right",
                    x=0.5,
                    y=1.1,
                    showactive=True
                )]
            )
        return fig
    
    # Chart type selection
    if st.button("Generate Visualization"):
        with st.spinner("Creating visualization..."):
            try:
                # Auto-detect visualization type if selected
                if viz_type == "Auto-detect":
                    if x_axis and y_axis:
                        if color_by and color_by in categorical_cols:
                            # Scatter plot for numerical vs numerical with color category
                            fig = px.scatter(
                                st.session_state.df, 
                                x=x_axis, 
                                y=y_axis, 
                                color=color_by,
                                hover_data=hover_data,
                                animation_frame=animation_frame,
                                title=f"{y_axis} vs {x_axis} by {color_by}",
                                template=color_theme.lower()
                            )
                        else:
                            # Simple scatter plot
                            fig = px.scatter(
                                st.session_state.df, 
                                x=x_axis, 
                                y=y_axis,
                                hover_data=hover_data,
                                animation_frame=animation_frame,
                                title=f"{y_axis} vs {x_axis}",
                                template=color_theme.lower()
                            )
                    elif x_axis and not y_axis:
                        # Histogram for single numerical column
                        fig = px.histogram(
                            st.session_state.df, 
                            x=x_axis, 
                            color=color_by,
                            animation_frame=animation_frame,
                            title=f"Distribution of {x_axis}",
                            template=color_theme.lower()
                        )
                
                # Manual visualization type selection
                elif viz_type == "Scatter Plot" and x_axis and y_axis:
                    fig = px.scatter(
                        st.session_state.df, 
                        x=x_axis, 
                        y=y_axis,
                        color=color_by,
                        hover_data=hover_data,
                        animation_frame=animation_frame,
                        title=f"Scatter Plot: {y_axis} vs {x_axis}",
                        template=color_theme.lower()
                    )
                    
                elif viz_type == "Treemap" and (x_axis or y_axis):
                    path = [col for col in [x_axis, y_axis, color_by, facet_col] if col]
                    if len(path) > 1:
                        fig = px.treemap(
                            st.session_state.df,
                            path=path,
                            values=numeric_cols[0] if numeric_cols else None,
                            title=f"Treemap of {', '.join(path)}",
                            template=color_theme.lower()
                        )
                    
                elif viz_type == "Sunburst" and (x_axis or y_axis):
                    path = [col for col in [x_axis, y_axis, color_by, facet_col] if col]
                    if len(path) > 1:
                        fig = px.sunburst(
                            st.session_state.df,
                            path=path,
                            values=numeric_cols[0] if numeric_cols else None,
                            title=f"Sunburst Chart of {', '.join(path)}",
                            template=color_theme.lower()
                        )
                        
                elif viz_type == "Parallel Coordinates" and len(numeric_cols) > 1:
                    dimensions = [{'label': col, 'values': st.session_state.df[col]} for col in numeric_cols[:5]]
                    fig = go.Figure(data=go.Parcoords(
                        line=dict(color=st.session_state.df[color_by].astype('category').cat.codes if color_by else None,
                               colorscale=px.colors.qualitative.Plotly),
                        dimensions=dimensions
                    ))
                    fig.update_layout(title="Parallel Coordinates Plot", template=color_theme.lower())
                    
                elif viz_type == "Parallel Categories" and len(categorical_cols) > 1:
                    dims = [st.session_state.df[col] for col in categorical_cols[:4]]
                    fig = px.parallel_categories(
                        st.session_state.df,
                        dimensions=dims,
                        color=st.session_state.df[numeric_cols[0]] if numeric_cols else None,
                        title="Parallel Categories Diagram",
                        template=color_theme.lower()
                    )
                    
                elif viz_type == "Density Contour" and x_axis and y_axis:
                    fig = px.density_contour(
                        st.session_state.df,
                        x=x_axis,
                        y=y_axis,
                        color=color_by,
                        marginal_x="histogram",
                        marginal_y="histogram",
                        title=f"Density Contour: {y_axis} vs {x_axis}",
                        template=color_theme.lower()
                    )
                    
                elif viz_type == "Dashboard View":
                    # Create a dashboard with multiple visualizations
                    st.subheader("Interactive Dashboard")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram
                        if x_axis:
                            hist_fig = px.histogram(
                                st.session_state.df,
                                x=x_axis,
                                color=color_by,
                                title=f"Distribution of {x_axis}",
                                template=color_theme.lower()
                            )
                            st.plotly_chart(hist_fig, use_container_width=True)
                        
                        # Box plot
                        if x_axis and y_axis:
                            box_fig = px.box(
                                st.session_state.df,
                                x=x_axis if x_axis in categorical_cols else y_axis,
                                y=y_axis if y_axis in numeric_cols else x_axis,
                                color=color_by,
                                title=f"Box Plot of {y_axis} by {x_axis}",
                                template=color_theme.lower()
                            )
                            st.plotly_chart(box_fig, use_container_width=True)
                    
                    with col2:
                        # Scatter plot
                        if x_axis and y_axis:
                            scatter_fig = px.scatter(
                                st.session_state.df,
                                x=x_axis,
                                y=y_axis,
                                color=color_by,
                                hover_data=hover_data,
                                title=f"{y_axis} vs {x_axis}",
                                template=color_theme.lower()
                            )
                            st.plotly_chart(scatter_fig, use_container_width=True)
                        
                        # Pie chart for categorical data
                        if x_axis and x_axis in categorical_cols:
                            pie_fig = px.pie(
                                st.session_state.df,
                                names=x_axis,
                                values=numeric_cols[0] if numeric_cols else None,
                                title=f"Distribution of {x_axis}",
                                template=color_theme.lower()
                            )
                            st.plotly_chart(pie_fig, use_container_width=True)
                    
                    st.markdown("---")
                    st.write("### Data Summary")
                    st.dataframe(st.session_state.df.describe())
                    pass  # Skip the rest of the visualization code for dashboard view
                    
                elif viz_type == "Line Chart" and x_axis and y_axis:
                    fig = px.line(
                        st.session_state.df, 
                        x=x_axis, 
                        y=y_axis,
                        color=color_by,
                        hover_data=hover_data,
                        title=f"Line Chart: {y_axis} over {x_axis}"
                    )
                
                elif viz_type == "Bar Chart" and x_axis and y_axis:
                    fig = px.bar(
                        st.session_state.df, 
                        x=x_axis, 
                        y=y_axis,
                        color=color_by,
                        hover_data=hover_data,
                        title=f"Bar Chart: {y_axis} by {x_axis}"
                    )
                
                elif viz_type == "Histogram" and x_axis:
                    fig = px.histogram(
                        st.session_state.df, 
                        x=x_axis,
                        color=color_by,
                        hover_data=hover_data,
                        title=f"Distribution of {x_axis}"
                    )
                
                elif viz_type == "Box Plot" and x_axis and y_axis:
                    fig = px.box(
                        st.session_state.df, 
                        x=x_axis, 
                        y=y_axis,
                        color=color_by,
                        hover_data=hover_data,
                        title=f"Box Plot: {y_axis} by {x_axis}"
                    )
                
                elif viz_type == "Violin Plot" and x_axis and y_axis:
                    fig = px.violin(
                        st.session_state.df, 
                        x=x_axis, 
                        y=y_axis,
                        color=color_by,
                        hover_data=hover_data,
                        title=f"Violin Plot: {y_axis} by {x_axis}"
                    )
                
                elif viz_type == "Heatmap" and numeric_cols:
                    corr = st.session_state.df[numeric_cols].corr()
                    fig = px.imshow(
                        corr, 
                        text_auto=True, 
                        aspect="auto",
                        title="Correlation Heatmap"
                    )
                
                elif viz_type == "Correlation Matrix" and numeric_cols:
                    fig = px.scatter_matrix(
                        st.session_state.df[numeric_cols],
                        title="Correlation Matrix"
                    )
                
                # Add animation if specified
                if fig is not None and animation_frame is not None:
                    try:
                        # Add animation frame to the figure if it's a Plotly Express figure
                        if hasattr(fig, 'data') and hasattr(fig, 'layout'):
                            # Update the animation frame for the figure
                            fig.update_layout(
                                updatemenus=[
                                    dict(
                                        type="buttons",
                                        buttons=[
                                            dict(
                                                label="Play",
                                                method="animate",
                                                args=[
                                                    None, 
                                                    {
                                                        "frame": {"duration": 1000, "redraw": True},
                                                        "fromcurrent": True, 
                                                        "transition": {"duration": 300}
                                                    }
                                                ]
                                            )
                                        ]
                                    )
                                ],
                                # Add slider for the animation
                                sliders=[{
                                    "active": 0,
                                    "yanchor": "top",
                                    "xanchor": "left",
                                    "currentvalue": {
                                        "font": {"size": 20},
                                        "prefix": f"{animation_frame}:",
                                        "visible": True,
                                        "xanchor": "right"
                                    },
                                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                                    "pad": {"b": 10, "t": 50},
                                    "len": 0.9,
                                    "x": 0.1,
                                    "y": 0
                                }]
                            )
                    except Exception as e:
                        st.warning(f"Could not add animation: {str(e)}")
                
                # Display the figure
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select appropriate columns for the visualization.")
                    
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                st.exception(e)
    
    # AI-Powered Visualization Suggestions
    st.subheader("AI-Powered Visualization Suggestions")
    if st.button("Get AI Visualization Suggestions"):
        with st.spinner("Analyzing data and generating visualization suggestions..."):
            try:
                # Initialize response_placeholder at the beginning
                response_placeholder = st.empty()
                full_response = ""
                
                # Prepare data description for the model
                data_desc = f"""
                DataFrame Info:
                - Shape: {st.session_state.df.shape}
                - Numeric Columns: {', '.join(numeric_cols) if numeric_cols else 'None'}
                - Categorical Columns: {', '.join(categorical_cols) if categorical_cols else 'None'}
                - Datetime Columns: {', '.join(datetime_cols) if datetime_cols else 'None'}
                - Sample Data (first 3 rows):
                {st.session_state.df.head(3).to_string()}
                """
                
                # Get visualization suggestion from the model
                prompt = f"""
                Based on the following data, suggest 2-3 insightful visualizations.
                For each suggestion, provide:
                1. A brief explanation of what insights it provides
                2. The Python code to generate it using Plotly Express
                
                {data_desc}
                
                Focus on creating visualizations that reveal interesting patterns, 
                correlations, or anomalies in the data.
                
                Format your response as markdown with code blocks for the Python code.
                """
                
                # Get the response from the query_agent
                response = query_agent(prompt, is_visualization=True, df=st.session_state.df)
                
                # Display the response
                st.markdown("### AI-Generated Visualization Suggestions")
                
                if isinstance(response, dict) and "error" in response:
                    st.error(f"Error generating visualization: {response['error']}")
                    full_response = f"Error: {response['error']}"
                elif isinstance(response, str):
                    full_response = response
                    
                    # Extract mentioned columns from the response
                    mentioned_columns = []
                    for col in st.session_state.df.columns:
                        if col.lower() in response.lower():
                            mentioned_columns.append(col)
                    
                    # For each mentioned column, add a frequency analysis
                    for col in mentioned_columns:
                        if col in st.session_state.df.columns:
                            try:
                                # Add a section for the analysis
                                full_response += f"\n\n### Analysis of '{col}' Distribution"
                                
                                # Add frequency table
                                freq = st.session_state.df[col].value_counts().sort_index()
                                full_response += f"\n\n**Frequency Table:**\n{freq.to_string()}"
                                
                                # Add visualization if it makes sense
                                if len(freq) > 1 and len(freq) < 20:  # Avoid too many categories
                                    try:
                                        fig = px.bar(
                                            x=freq.index.astype(str), 
                                            y=freq.values,
                                            title=f"Distribution of {col}",
                                            labels={'x': col, 'y': 'Count'}
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        full_response += "\n\n*Visualization of the distribution is shown above.*"
                                    except Exception as viz_error:
                                        st.warning(f"Could not generate visualization for {col}: {str(viz_error)}")
                                
                                # Add basic statistics for numeric columns
                                if pd.api.types.is_numeric_dtype(st.session_state.df[col]):
                                    try:
                                        stats = st.session_state.df[col].describe()
                                        full_response += f"\n\n**Statistics:**\n{stats.to_string()}"
                                    except Exception as stats_error:
                                        st.warning(f"Could not generate statistics for {col}: {str(stats_error)}")
                            except Exception as col_error:
                                st.warning(f"Error processing column {col}: {str(col_error)}")
                
                # Display the response
                response_placeholder.markdown(full_response)
                
                # Add a toggle for showing the code
                if 'df' in st.session_state and st.session_state.df is not None:
                    with st.expander("View Analysis Code"):
                        # Generate the code for the analysis
                        code = "# Import required libraries\n"
                        code += "import pandas as pd\n"
                        code += "import plotly.express as px\n\n"
                        # Add code for frequency analysis of mentioned columns
                        for col in mentioned_columns:
                            if col in st.session_state.df.columns:
                                code += f"# Frequency analysis for '{col}'\n"
                                code += f"freq = df['{col}'].value_counts().sort_index()\n"
                                if pd.api.types.is_numeric_dtype(st.session_state.df[col]):
                                    code += f"stats = df['{col}'].describe()\n"
                                    code += f"print(f\"Statistics for {col}:\\n{{stats.to_string()}}\")\n\n"
                                
                                code += f"fig = px.bar(x=freq.index.astype(str), y=freq.values, \n                    title='Distribution of {col}',\n                    labels={{'x': '{col}', 'y': 'Count'}})\n"
                                code += "fig.show()\n\n"
                        
                        st.code(code, language='python')
                        
                        if st.button("Run This Code"):
                            try:
                                # Create a local copy of the dataframe
                                local_vars = {'df': st.session_state.df.copy()}
                                
                                # Execute the code in a safe environment
                                exec(code, globals(), local_vars)
                                
                                # Show success message
                                st.success("Code executed successfully!")
                                
                            except Exception as e:
                                st.error(f"Error executing code: {str(e)}")
            
            except Exception as e:
                st.error(f"An error occurred while generating the response: {str(e)}")
                st.exception(e)
                # The code generation and execution has been moved to the main try block
                # to avoid duplicate code and improve maintainability
                pass
            
            # Extract and display any code blocks in the response
            code_blocks = extract_code_blocks(full_response)
            for i, code in enumerate(code_blocks):
                with st.expander(f"View and Run Suggestion #{i+1}"):
                    st.code(code, language="python")
                    
                    if st.button(f"Run Suggestion #{i+1}", key=f"run_suggestion_{i}"):
                        try:
                            # Create a safe environment for code execution with necessary imports
                            safe_globals = {
                                '__builtins__': {
                                    'print': print,
                                    'len': len,
                                    'str': str,
                                    'int': int,
                                    'float': float,
                                    'list': list,
                                    'dict': dict,
                                    'set': set,
                                    'tuple': tuple,
                                    'range': range,
                                    'enumerate': enumerate,
                                    'zip': zip,
                                    'sorted': sorted,
                                    'isinstance': isinstance,
                                    'type': type,
                                    'sum': sum,
                                    'min': min,
                                    'max': max,
                                    'abs': abs,
                                    'round': round,
                                    'bool': bool
                                },
                                'pd': pd,
                                'np': np,
                                'px': px,
                                'go': go,
                                'plt': plt,
                                'df': st.session_state.df.copy()
                            }
                            
                            # Add any other necessary imports
                            exec('import pandas as pd', safe_globals)
                            exec('import numpy as np', safe_globals)
                            exec('import plotly.express as px', safe_globals)
                            exec('import plotly.graph_objects as go', safe_globals)
                            exec('import matplotlib.pyplot as plt', safe_globals)
                            
                            # Execute the user's code
                            try:
                                # First try to compile the code to check for syntax errors
                                compiled_code = compile(code, '<string>', 'exec')
                                # Then execute it in the safe environment
                                exec(compiled_code, safe_globals)
                            except SyntaxError as se:
                                st.error(f"Syntax error in the code: {str(se)}")
                                st.code(f"Error on line {se.lineno}: {se.text}{' ' * (se.offset-1)}^", language='python')
                                raise
                            except Exception as e:
                                st.error(f"Error executing the code: {str(e)}")
                                st.exception(e)
                                raise
                            finally:
                                # Clean up any resources if needed
                                pass
                            
                            # Try to get the figure from different possible variable names
                            fig = None
                            for var_name, var_value in safe_globals.items():
                                if var_name.startswith('_') or var_name in ['pd', 'np', 'px', 'go', 'plt', 'df']:
                                    continue
                                if isinstance(var_value, (go.Figure, type(px.scatter(pd.DataFrame())))):
                                    fig = var_value
                                    break
                            
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                                st.success("✅ Visualization generated successfully!")
                            else:
                                # Check if any figures were created with plt
                                if 'plt' in safe_globals and hasattr(safe_globals['plt'], 'gcf'):
                                    fig = safe_globals['plt'].gcf()
                                    if fig.get_axes():
                                        st.pyplot(fig)
                                        st.success("✅ Matplotlib visualization generated successfully!")
                                    else:
                                        st.warning("No figure was generated. Make sure the code creates a Plotly or Matplotlib figure.")
                                else:
                                    st.warning("No figure was generated. Make sure the code creates a Plotly or Matplotlib figure.")
                            
                        except Exception as e:
                            st.error(f"Error executing visualization code: {str(e)}")
                            st.warning("Please try a different visualization or check your code for errors.")
                            st.code(code, language='python')
                        else:
                            st.warning("No code blocks found in the response.")
                    else:
                        st.warning("Unexpected response format from the AI model.")
            
            # except Exception as e:
            #     st.error(f"Error generating visualization: {str(e)}")
    
    # ==================== Statistical Analysis ====================
    st.markdown("---")
    st.subheader("📈 Statistical Analysis")
    
    stat_tabs = st.tabs(["Descriptive Stats", "Statistical Tests", "Correlation Analysis", "Feature Importance", "Statistical Modeling"])
    
    with stat_tabs[0]:  # Descriptive Stats
        st.subheader("Descriptive Statistics")
        
        # Basic statistics
        st.write("### Basic Statistics")
        st.dataframe(st.session_state.df.describe(include='all').T)
        
        # Missing values
        st.write("### Missing Values")
        missing_df = pd.DataFrame({
            'Missing Values': st.session_state.df.isnull().sum(),
            'Percentage': (st.session_state.df.isnull().sum() / len(st.session_state.df)) * 100
        })
        st.dataframe(missing_df[missing_df['Missing Values'] > 0])
        
        # Data types
        st.write("### Data Types")
        st.write(pd.DataFrame(st.session_state.df.dtypes, columns=['Data Type']))
        
    with stat_tabs[1]:  # Statistical Tests
        st.subheader("Statistical Tests")
        
        # T-test
        st.write("### T-Test (Independent Samples)")
        col1, col2 = st.columns(2)
        with col1:
            num_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            ttest_var = st.selectbox("Select numeric variable", num_cols)
        with col2:
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            ttest_group = st.selectbox("Select categorical variable with 2 groups", cat_cols)
            
        if st.button("Run T-Test"):
            try:
                groups = st.session_state.df[ttest_group].dropna().unique()
                if len(groups) == 2:
                    group1 = st.session_state.df[st.session_state.df[ttest_group] == groups[0]][ttest_var].dropna()
                    group2 = st.session_state.df[st.session_state.df[ttest_group] == groups[1]][ttest_var].dropna()
                    t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
                    
                    st.write(f"**T-Test Results for {ttest_var} by {ttest_group}**")
                    st.write(f"T-statistic: {t_stat:.4f}")
                    st.write(f"P-value: {p_val:.4f}")
                    st.write("Significant at p < 0.05:", "✅" if p_val < 0.05 else "❌")
                else:
                    st.warning("Selected categorical variable must have exactly 2 groups")
            except Exception as e:
                st.error(f"Error performing t-test: {str(e)}")
        
        # ANOVA
        st.write("### One-Way ANOVA")
        col1, col2 = st.columns(2)
        with col1:
            anova_var = st.selectbox("Select numeric variable", num_cols, key="anova_num")
        with col2:
            anova_group = st.selectbox("Select categorical variable", cat_cols, key="anova_cat")
            
        if st.button("Run ANOVA"):
            try:
                groups = st.session_state.df[anova_group].dropna().unique()
                if len(groups) > 1:
                    group_data = [st.session_state.df[st.session_state.df[anova_group] == group][anova_var].dropna() 
                                for group in groups]
                    f_stat, p_val = f_oneway(*group_data)
                    
                    st.write(f"**ANOVA Results for {anova_var} by {anova_group}**")
                    st.write(f"F-statistic: {f_stat:.4f}")
                    st.write(f"P-value: {p_val:.4f}")
                    st.write("Significant at p < 0.05:", "✅" if p_val < 0.05 else "❌")
                else:
                    st.warning("Selected categorical variable must have at least 2 groups")
            except Exception as e:
                st.error(f"Error performing ANOVA: {str(e)}")
    
    with stat_tabs[2]:  # Correlation Analysis
        st.subheader("Correlation Analysis")
        
        # Correlation matrix
        st.write("### Correlation Matrix")
        corr_method = st.selectbox("Correlation method", 
                                 ["pearson", "spearman", "kendall"])
        
        try:
            corr_matrix = st.session_state.df.select_dtypes(include=['number']).corr(method=corr_method)
            fig = px.imshow(corr_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          color_continuous_scale='RdBu_r',
                          zmin=-1, 
                          zmax=1)
            st.plotly_chart(fig, use_container_width=True)
            
            # Get top correlations
            st.write("### Top Correlations")
            corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
            corr_pairs = corr_pairs[corr_pairs != 1].drop_duplicates()
            st.dataframe(corr_pairs.head(10).reset_index().rename(columns={0: 'Correlation'}))
            
        except Exception as e:
            st.error(f"Error calculating correlations: {str(e)}")
    
    with stat_tabs[3]:  # Feature Importance
        st.subheader("Feature Importance Analysis")
        
        target_col = st.selectbox("Select target variable", 
                                st.session_state.df.select_dtypes(include=['number', 'category', 'object']).columns.tolist())
        
        if st.button("Calculate Feature Importance"):
            try:
                # Prepare data
                df_clean = st.session_state.df.copy()
                
                # Encode categorical variables
                le = LabelEncoder()
                for col in df_clean.select_dtypes(include=['object', 'category']).columns:
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                
                # Handle missing values
                df_clean = df_clean.fillna(df_clean.median())
                
                X = df_clean.drop(columns=[target_col])
                y = df_clean[target_col]
                
                # Choose appropriate model based on target type
                if pd.api.types.is_numeric_dtype(y):
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    importance = mutual_info_regression(X, y, random_state=42)
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    importance = mutual_info_classif(X, y, random_state=42)
                
                # Fit model
                model.fit(X, y)
                
                # Get feature importances
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_,
                    'Mutual_Info': importance
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importance
                fig = px.bar(feature_importance, 
                            x='Importance', 
                            y='Feature', 
                            orientation='h',
                            title='Feature Importance')
                st.plotly_chart(fig, use_container_width=True)
                
                # Show mutual information
                fig_mi = px.bar(feature_importance, 
                              x='Mutual_Info', 
                              y='Feature', 
                              orientation='h',
                              title='Mutual Information')
                st.plotly_chart(fig_mi, use_container_width=True)
                
                # Show importance table
                st.dataframe(feature_importance)
                
            except Exception as e:
                st.error(f"Error calculating feature importance: {str(e)}")
    
    with stat_tabs[4]:  # Statistical Modeling
        st.subheader("Statistical Modeling")
        
        model_type = st.selectbox("Select model type", 
                                ["Linear Regression", "Logistic Regression"])
        
        if model_type == "Linear Regression":
            st.write("### Linear Regression")
            col1, col2 = st.columns(2)
            with col1:
                target = st.selectbox("Select target variable", 
                                    st.session_state.df.select_dtypes(include=['number']).columns.tolist())
            with col2:
                features = st.multiselect("Select features", 
                                        st.session_state.df.select_dtypes(include=['number']).columns.tolist())
            
            if st.button("Run Linear Regression"):
                try:
                    df_clean = st.session_state.df[[target] + features].dropna()
                    X = sm.add_constant(df_clean[features])
                    y = df_clean[target]
                    
                    model = sm.OLS(y, X).fit()
                    
                    st.write("### Model Summary")
                    st.text(str(model.summary()))
                    
                    # Plot actual vs predicted
                    df_clean['Predicted'] = model.predict(X)
                    fig = px.scatter(df_clean, x=target, y='Predicted', 
                                   title='Actual vs Predicted Values',
                                   trendline='ols')
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error running linear regression: {str(e)}")
        
        elif model_type == "Logistic Regression":
            st.write("### Logistic Regression")
            col1, col2 = st.columns(2)
            with col1:
                target = st.selectbox("Select binary target variable", 
                                    st.session_state.df.select_dtypes(include=['category', 'object']).columns.tolist())
            with col2:
                features = st.multiselect("Select features", 
                                        st.session_state.df.select_dtypes(include=['number']).columns.tolist())
            
            if st.button("Run Logistic Regression"):
                try:
                    df_clean = st.session_state.df[[target] + features].dropna()
                    X = sm.add_constant(df_clean[features])
                    y = pd.factorize(df_clean[target])[0]  # Convert to numeric
                    
                    model = sm.Logit(y, X).fit(disp=0)
                    
                    st.write("### Model Summary")
                    st.text(str(model.summary()))
                    
                    # Plot ROC curve
                    from sklearn.metrics import roc_curve, roc_auc_score
                    y_pred_proba = model.predict(X)
                    fpr, tpr, _ = roc_curve(y, y_pred_proba)
                    auc = roc_auc_score(y, y_pred_proba)
                    
                    fig = px.area(x=fpr, y=tpr, 
                                title=f'ROC Curve (AUC = {auc:.2f})',
                                labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
                    fig.add_shape(type='line', line=dict(dash='dash'),
                                x0=0, x1=1, y0=0, y1=1)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error running logistic regression: {str(e)}")
    # Add a section for custom visualization code
    with st.expander("Advanced: Custom Visualization Code"):
        custom_code = st.text_area("Enter your custom visualization code (using Plotly Express):", 
                                 height=200,
                                 value="""import plotly.express as px
# Example: fig = px.scatter(df, x='column1', y='column2', color='category')
# Make sure to assign your figure to 'fig'
""")

    # Move the rest of the code into the appropriate tabs
    if 'df' in st.session_state and st.session_state.df is not None:
        with tab1:
            # Move existing data analysis code here
            pass
            
        with tab2:
            # Move existing visualization code here
            pass
        
        if st.button("Run Custom Code"):
            try:
                local_vars = {
                    'df': st.session_state.df.copy(),
                    'px': px,
                    'pd': pd,
                    'np': np
                }
                
                # Create a safe globals dictionary with necessary builtins and imports
                safe_globals = {
                    '__builtins__': {
                        '__import__': __import__,
                        'print': print,
                        'range': range,
                        'len': len,
                        'list': list,
                        'dict': dict,
                        'str': str,
                        'int': int,
                        'float': float,
                        'bool': bool,
                        'type': type,
                        'isinstance': isinstance,
                        'Exception': Exception
                    },
                    'px': px,
                    'pd': pd,
                    'np': np,
                    'go': go,
                    'plt': plt
                }
                safe_globals.update(local_vars)
                
                # Execute the code with the safe globals
                exec(custom_code, safe_globals, local_vars)
                
                fig = local_vars.get('fig')
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("Custom visualization generated successfully!")
                else:
                    st.warning("No figure was created. Make sure to assign your plot to 'fig'.")
                    
            except Exception as e:
                st.error(f"Error executing custom code: {str(e)}")
                st.exception(e) 

# Multimodal input (optional)
st.sidebar.markdown("---")
st.sidebar.subheader("Multimodal (Image/Sketch)")
image_file = st.sidebar.file_uploader("Upload an image/sketch (optional)", type=["png", "jpg", "jpeg"])
VISION_MODEL = "togethercomputer/Llama-3-8B-Vision-Instruct"
if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    if st.button("Describe this image/sketch"):
        # Read image bytes
        image_bytes = image_file.read()
        vision_api_url = "https://api.together.xyz/v1/vision/generate"
        headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
        files = {"file": (image_file.name, image_bytes, image_file.type)}
        data = {"model": VISION_MODEL, "prompt": "Describe this image/sketch for data analysis."}
        try:
            resp = requests.post(vision_api_url, headers=headers, files=files, data=data, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            vision_response = result.get("output", "[No description returned]")
        except Exception as e:
            vision_response = f"[Vision model error: {e}]"
        st.markdown(f"**Vision Model:** {vision_response}") 