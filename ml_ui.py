"""
Machine Learning UI Components for Streamlit

This module provides UI components for the enhanced ML features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import shap
from ml_enhancements import (
    AutomatedFeatureEngineering, ModelComparator, ModelInterpreter,
    UnsupervisedLearning, get_available_models, create_preprocessing_pipeline,
    hyperparameter_tuning
)

class MLUI:
    """Streamlit UI components for machine learning"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize ML UI components
        
        Args:
            df: Input DataFrame
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.models = {}
        self.task = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.model_comparator = None
        self.best_model = None
        
    def select_task(self):
        """Select ML task (regression or classification)"""
        st.sidebar.subheader("Machine Learning Task")
        self.task = st.sidebar.radio(
            "Select Task",
            ["Regression", "Classification"],
            index=0,
            key=f"{id(self)}_ml_task_radio"  # Unique key using object ID
        ).lower()
        
        # Select target variable with unique key
        target_col = st.sidebar.selectbox(
            "Select Target Variable",
            self.df.columns,
            key=f"{id(self)}_target_col"  # Unique key using object ID
        )
        
        # Select features with unique key
        feature_cols = st.sidebar.multiselect(
            "Select Features",
            [col for col in self.df.columns if col != target_col],
            default=[col for col in self.numeric_cols if col != target_col],
            key=f"{id(self)}_feature_cols"  # Unique key using object ID
        )
        
        # Split data with unique key
        test_size = st.sidebar.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key=f"{id(self)}_test_size"  # Unique key using object ID
        )
        
        # Random state with unique key
        random_state = st.sidebar.number_input(
            "Random State",
            min_value=0,
            value=42,
            step=1,
            key=f"{id(self)}_random_state"  # Unique key using object ID
        )
        
        return target_col, feature_cols, test_size, random_state
    
    def feature_engineering_ui(self):
        """UI for feature engineering options"""
        st.sidebar.subheader("Feature Engineering")
        
        col1, col2 = st.sidebar.columns(2)
        
        # Add unique keys to all interactive elements
        create_interactions = col1.checkbox(
            "Create Interactions", 
            value=True,
            key=f"{id(self)}_create_interactions"
        )
        create_polynomials = col1.checkbox(
            "Create Polynomials", 
            value=True,
            key=f"{id(self)}_create_polynomials"
        )
        poly_degree = col2.number_input(
            "Polynomial Degree",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            key=f"{id(self)}_poly_degree"
        )
        create_stats = col2.checkbox(
            "Statistical Features", 
            value=True,
            key=f"{id(self)}_create_stats"
        )
        
        return {
            'create_interactions': create_interactions,
            'create_polynomials': create_polynomials,
            'polynomial_degree': poly_degree,
            'create_statistical_features': create_stats
        }
    
    def preprocess_data(self, target_col: str, feature_cols: List[str], 
                       test_size: float, random_state: int, 
                       feature_eng_params: Dict):
        """
        Preprocess data for modeling
        
        Args:
            target_col: Name of target column
            feature_cols: List of feature columns
            test_size: Size of test set
            random_state: Random seed
            feature_eng_params: Feature engineering parameters
            
        Returns:
            Preprocessed data
        """
        from sklearn.model_selection import train_test_split
        
        # Separate features and target
        X = self.df[feature_cols].copy()
        y = self.df[target_col]
        
        # Apply feature engineering
        if any(feature_eng_params.values()):
            fe = AutomatedFeatureEngineering(**feature_eng_params)
            X = fe.fit_transform(X)
        
        # Identify numeric and categorical features after feature engineering
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Fit preprocessor on training data and transform both sets
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        
        # Get feature names after one-hot encoding
        if hasattr(preprocessor.named_transformers_['cat'], 'named_steps'):
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            if hasattr(ohe, 'get_feature_names_out'):
                ohe_feature_names = ohe.get_feature_names_out(categorical_features)
                all_feature_names = numeric_features + list(ohe_feature_names)
                self.feature_names = all_feature_names
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def model_selection_ui(self):
        """UI for model selection"""
        st.subheader("Model Selection")
        
        # Get available models for the task
        available_models = get_available_models(self.task)
        
        # Let user select models
        selected_models = st.multiselect(
            "Select Models to Compare",
            list(available_models.keys()),
            default=["Random Forest", "XGBoost", "LightGBM"]
        )
        
        # Create model instances
        self.models = {name: available_models[name] for name in selected_models}
        
        # Add hyperparameter tuning options
        st.subheader("Hyperparameter Tuning")
        tune_hyperparams = st.checkbox("Enable Hyperparameter Tuning", value=False)
        
        param_grids = {}
        if tune_hyperparams:
            st.write("Configure hyperparameter search space")
            
            for name in selected_models:
                with st.expander(f"{name} Parameters"):
                    if "Random Forest" in name:
                        n_estimators = st.text_input(
                            "Number of estimators", 
                            value="100,200,300",
                            key=f"n_est_{name}"
                        )
                        max_depth = st.text_input(
                            "Max depth", 
                            value="None,5,10",
                            key=f"max_depth_{name}"
                        )
                        param_grids[name] = {
                            'n_estimators': [int(x) if x != 'None' else None 
                                           for x in n_estimators.split(',')],
                            'max_depth': [int(x) if x != 'None' else None 
                                         for x in max_depth.split(',')]
                        }
                    elif "XGBoost" in name or "LightGBM" in name:
                        learning_rate = st.text_input(
                            "Learning rate", 
                            value="0.01,0.1,0.3",
                            key=f"lr_{name}"
                        )
                        n_estimators = st.text_input(
                            "Number of estimators", 
                            value="100,200",
                            key=f"n_est_{name}"
                        )
                        param_grids[name] = {
                            'learning_rate': [float(x) for x in learning_rate.split(',')],
                            'n_estimators': [int(x) for x in n_estimators.split(',')]
                        }
                    # Add more model-specific parameters as needed
        
        return param_grids
    
    def train_and_compare_models(self, param_grids=None):
        """
        Train and compare multiple models
        
        Args:
            param_grids: Dictionary of parameter grids for hyperparameter tuning
        """
        if not hasattr(self, 'X_train') or self.X_train is None:
            st.warning("Please preprocess the data first.")
            return
        
        # Initialize model comparator
        scoring = 'r2' if self.task == 'regression' else 'accuracy'
        self.model_comparator = ModelComparator(self.models, task=self.task, metric=scoring)
        
        # Train and compare models
        with st.spinner("Training and comparing models..."):
            results = self.model_comparator.compare_models(
                self.X_train, 
                self.y_train,
                params=param_grids
            )
            
            # Display results
            st.subheader("Model Comparison Results")
            st.dataframe(results.style.highlight_max(axis=0))
            
            # Plot comparison
            st.subheader("Model Performance Comparison")
            fig = self.model_comparator.plot_comparison()
            st.pyplot(fig)
            
            # Get best model
            self.best_model_name, self.best_model = self.model_comparator.get_best_model()
            st.success(f"Best model: {self.best_model_name}")
    
    def model_interpretation_ui(self):
        """UI for model interpretation"""
        if not hasattr(self, 'best_model') or self.best_model is None:
            st.warning("Please train and compare models first.")
            return
        
        st.subheader("Model Interpretation")
        
        # SHAP summary plot
        st.write("### SHAP Feature Importance")
        if st.button("Generate SHAP Summary Plot"):
            with st.spinner("Generating SHAP summary plot..."):
                try:
                    # Create SHAP explainer
                    explainer = shap.Explainer(self.best_model, self.X_train, feature_names=self.feature_names)
                    shap_values = explainer(self.X_test)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating SHAP plot: {str(e)}")
        
        # LIME explanation for a specific instance
        st.write("### LIME Explanation")
        instance_idx = st.number_input(
            "Select instance index for explanation",
            min_value=0,
            max_value=len(self.X_test)-1,
            value=0,
            step=1
        )
        
        if st.button("Generate LIME Explanation"):
            with st.spinner("Generating LIME explanation..."):
                try:
                    # Create LIME explainer
                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        self.X_train,
                        feature_names=self.feature_names,
                        class_names=['target'],
                        mode='regression' if self.task == 'regression' else 'classification'
                    )
                    
                    # Select instance
                    instance = self.X_test[instance_idx]
                    
                    # Get explanation
                    exp = explainer.explain_instance(
                        instance,
                        self.best_model.predict,
                        num_features=10
                    )
                    
                    # Display explanation
                    st.write(exp.as_list())
                    
                    # Plot explanation
                    fig = exp.as_pyplot_figure()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating LIME explanation: {str(e)}")
    
    def unsupervised_learning_ui(self):
        """UI for unsupervised learning"""
        st.subheader("Unsupervised Learning")
        
        # Dimensionality reduction
        st.write("### Dimensionality Reduction")
        max_components = max(2, min(10, len(self.numeric_cols)))
        n_components = st.slider(
            "Number of components",
            min_value=1,
            max_value=max_components,
            value=min(2, max_components),
            step=1
        )
        
        if st.button("Perform PCA"):
            with st.spinner("Performing PCA..."):
                try:
                    # Select numeric features
                    X = self.df[self.numeric_cols].dropna()
                    
                    # Perform PCA
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=n_components)
                    X_pca = pca.fit_transform(X)
                    
                    # Plot explained variance ratio
                    fig1, ax1 = plt.subplots()
                    ax1.plot(range(1, n_components+1), np.cumsum(pca.explained_variance_ratio_), 'bo-')
                    ax1.set_xlabel('Number of Components')
                    ax1.set_ylabel('Cumulative Explained Variance Ratio')
                    ax1.set_title('Explained Variance Ratio')
                    st.pyplot(fig1)
                    
                    # Plot first two components
                    if n_components >= 2:
                        fig2, ax2 = plt.subplots()
                        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
                        ax2.set_xlabel('First Principal Component')
                        ax2.set_ylabel('Second Principal Component')
                        ax2.set_title('First Two Principal Components')
                        st.pyplot(fig2)
                except Exception as e:
                    st.error(f"Error performing PCA: {str(e)}")
        
        # Clustering
        st.write("### Clustering")
        n_clusters = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=10,
            value=3,
            step=1
        )
        
        if st.button("Perform Clustering"):
            with st.spinner("Performing clustering..."):
                try:
                    # Select numeric features
                    X = self.df[self.numeric_cols].dropna()
                    
                    # Perform K-means clustering
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X)
                    
                    # Create a new DataFrame with only the data we want to display
                    df_clustered = pd.DataFrame(index=X.index)
                    
                    # Convert numeric columns to float with 2 decimal places for display
                    for col in X.select_dtypes(include=['number']).columns:
                        df_clustered[col] = X[col].round(2)
                    
                    # Add cluster information
                    df_clustered['Cluster'] = clusters.astype(str)
                    
                    # Convert all columns to string for consistent display
                    df_clustered = df_clustered.astype(str)
                    
                    # Plot clusters (first two features)
                    if len(X.columns) >= 2:
                        fig, ax = plt.subplots()
                        scatter = ax.scatter(
                            X.iloc[:, 0], 
                            X.iloc[:, 1], 
                            c=clusters, 
                            cmap='viridis',
                            alpha=0.7
                        )
                        ax.set_xlabel(X.columns[0])
                        ax.set_ylabel(X.columns[1])
                        ax.set_title('Clustering Results')
                        plt.colorbar(scatter, label='Cluster')
                        st.pyplot(fig)
                    
                    # Show cluster statistics
                    st.write("### Cluster Statistics")
                    cluster_stats = df_clustered.groupby('Cluster').agg(['mean', 'std'])
                    st.dataframe(cluster_stats)
                    
                except Exception as e:
                    st.error(f"Error performing clustering: {str(e)}")

def main():
    """Main function for testing the ML UI"""
    st.title("Machine Learning Workbench")
    
    # Load sample data
    from sklearn.datasets import load_iris, load_diabetes
    
    dataset = st.selectbox(
        "Select Dataset",
        ["Iris (Classification)", "Diabetes (Regression)"]
    )
    
    if dataset == "Iris (Classification)":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    else:  # Diabetes
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    
    # Initialize ML UI
    ml_ui = MLUI(df)
    
    # Display data
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # Task selection
    target_col, feature_cols, test_size, random_state = ml_ui.select_task()
    
    # Feature engineering
    feature_eng_params = ml_ui.feature_engineering_ui()
    
    # Preprocess data
    if st.sidebar.button("Preprocess Data"):
        with st.spinner("Preprocessing data..."):
            try:
                ml_ui.preprocess_data(
                    target_col, 
                    feature_cols, 
                    test_size, 
                    random_state,
                    feature_eng_params
                )
                st.success("Data preprocessed successfully!")
            except Exception as e:
                st.error(f"Error during preprocessing: {str(e)}")
    
    # Model selection and training
    if hasattr(ml_ui, 'X_train') and ml_ui.X_train is not None:
        param_grids = ml_ui.model_selection_ui()
        
        if st.button("Train and Compare Models"):
            ml_ui.train_and_compare_models(param_grids if param_grids else None)
        
        # Model interpretation
        if hasattr(ml_ui, 'best_model') and ml_ui.best_model is not None:
            ml_ui.model_interpretation_ui()
    
    # Unsupervised learning
    ml_ui.unsupervised_learning_ui()

if __name__ == "__main__":
    main()
