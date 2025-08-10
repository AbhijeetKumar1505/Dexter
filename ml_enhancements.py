"""
Machine Learning Enhancements Module

This module extends the existing ML capabilities with:
- Advanced model selection
- Model comparison functionality
- Automated feature engineering
- Model interpretability tools (SHAP, LIME)
- Support for unsupervised learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
)
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, RidgeClassifier
)
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import shap
import lime
import lime.lime_tabular
from scipy import stats
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class AutomatedFeatureEngineering(BaseEstimator, TransformerMixin):
    """Automated feature engineering for tabular data"""
    
    def __init__(self, 
                 create_interactions: bool = True,
                 create_polynomials: bool = True,
                 polynomial_degree: int = 2,
                 create_statistical_features: bool = True):
        self.create_interactions = create_interactions
        self.create_polynomials = create_polynomials
        self.polynomial_degree = polynomial_degree
        self.create_statistical_features = create_statistical_features
        self.numeric_cols = None
        
    def fit(self, X, y=None):
        # Identify numeric columns
        if isinstance(X, pd.DataFrame):
            self.numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_transformed = X.copy()
        
        # Create interaction terms
        if self.create_interactions and len(self.numeric_cols) > 1:
            for i in range(len(self.numeric_cols)):
                for j in range(i+1, len(self.numeric_cols)):
                    col1, col2 = self.numeric_cols[i], self.numeric_cols[j]
                    X_transformed[f"{col1}_x_{col2}"] = X[col1] * X[col2]
        
        # Create polynomial features
        if self.create_polynomials and self.polynomial_degree > 1:
            for col in self.numeric_cols:
                for degree in range(2, self.polynomial_degree + 1):
                    X_transformed[f"{col}_pow_{degree}"] = X[col] ** degree
        
        # Create statistical features
        if self.create_statistical_features and len(self.numeric_cols) > 1:
            X_transformed['mean'] = X[self.numeric_cols].mean(axis=1)
            X_transformed['std'] = X[self.numeric_cols].std(axis=1)
            X_transformed['min'] = X[self.numeric_cols].min(axis=1)
            X_transformed['max'] = X[self.numeric_cols].max(axis=1)
        
        return X_transformed


class ModelComparator:
    """Compare multiple machine learning models"""
    
    def __init__(self, models: Dict[str, object], metric: str = None, 
                 task: str = 'regression', cv: int = 5):
        """
        Initialize the model comparator
        
        Args:
            models: Dictionary of model names and instances
            metric: Scoring metric (default: 'r2' for regression, 'accuracy' for classification)
            task: 'regression' or 'classification'
            cv: Number of cross-validation folds
        """
        self.models = models
        self.task = task
        self.cv = cv
        
        # Set default metrics if not provided
        if metric is None:
            self.metric = 'r2' if task == 'regression' else 'accuracy'
        else:
            self.metric = metric
            
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf
    
    def compare_models(self, X, y, params: Dict = None):
        """
        Compare models using cross-validation
        
        Args:
            X: Features
            y: Target
            params: Optional dictionary of parameters for each model
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for name, model in self.models.items():
            # Get parameters if provided
            model_params = params.get(name, {}) if params else {}
            
            # Clone the model and set parameters
            current_model = clone(model)
            if model_params:
                current_model.set_params(**model_params)
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                current_model, X, y, 
                cv=self.cv, 
                scoring=self.metric,
                n_jobs=-1
            )
            
            # Store results
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            self.results[name] = {
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_scores': cv_scores,
                'model': current_model
            }
            
            # Update best model
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_model = name
                
            results.append({
                'Model': name,
                f'Mean {self.metric}': mean_score,
                f'Std {self.metric}': std_score
            })
        
        return pd.DataFrame(results).sort_values(by=f'Mean {self.metric}', ascending=False)
    
    def get_best_model(self):
        """Get the best performing model"""
        if not self.results:
            raise ValueError("No models have been compared yet. Call compare_models() first.")
        return self.best_model, self.results[self.best_model]['model']
    
    def plot_comparison(self):
        """Plot comparison of model performances"""
        if not self.results:
            raise ValueError("No models have been compared yet. Call compare_models() first.")
            
        names = list(self.results.keys())
        means = [self.results[name]['mean_score'] for name in names]
        stds = [self.results[name]['std_score'] for name in names]
        
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(names))
        
        plt.barh(y_pos, means, xerr=stds, alpha=0.7, color='skyblue')
        plt.yticks(y_pos, names)
        plt.xlabel(f'Performance ({self.metric})')
        plt.title('Model Comparison')
        plt.tight_layout()
        
        return plt.gcf()


class ModelInterpreter:
    """Model interpretation using SHAP and LIME"""
    
    def __init__(self, model, X_train, feature_names=None, task='regression'):
        """
        Initialize the model interpreter
        
        Args:
            model: Trained model
            X_train: Training data (for LIME explainer)
            feature_names: List of feature names
            task: 'regression' or 'classification'
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.task = task
        self.explainer_shap = None
        self.explainer_lime = None
        
    def shap_summary_plot(self, X, plot_type='dot'):
        """
        Create a SHAP summary plot
        
        Args:
            X: Data to explain
            plot_type: Type of plot ('dot', 'bar', 'violin')
            
        Returns:
            Matplotlib figure
        """
        # Create SHAP explainer based on model type
        if 'tree' in str(type(self.model)).lower():
            explainer = shap.TreeExplainer(self.model)
        else:
            explainer = shap.KernelExplainer(self.model.predict, self.X_train)
            
        shap_values = explainer.shap_values(X)
        
        # For classification models with multiple classes
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
            
        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=self.feature_names,
            plot_type=plot_type,
            show=False
        )
        
        return plt.gcf()
    
    def lime_explanation(self, instance, num_features=5):
        """
        Generate LIME explanation for a single instance
        
        Args:
            instance: Instance to explain
            num_features: Number of features to show in explanation
            
        Returns:
            LIME explanation object and figure
        """
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            class_names=['target'],
            mode='regression' if self.task == 'regression' else 'classification'
        )
        
        # Generate explanation
        exp = explainer.explain_instance(
            instance,
            self.model.predict,
            num_features=num_features
        )
        
        # Create figure
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        
        return exp, fig


class UnsupervisedLearning:
    """Unsupervised learning methods"""
    
    def __init__(self, n_clusters=3, random_state=42):
        """
        Initialize unsupervised learning methods
        
        Args:
            n_clusters: Number of clusters for clustering algorithms
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        
    def perform_pca(self, X, n_components=2):
        """
        Perform Principal Component Analysis
        
        Args:
            X: Input data
            n_components: Number of principal components to keep
            
        Returns:
            Transformed data and explained variance ratio
        """
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        
        return X_pca, pca.explained_variance_ratio_
    
    def perform_clustering(self, X, method='kmeans', n_clusters=None):
        """
        Perform clustering
        
        Args:
            X: Input data
            method: 'kmeans', 'dbscan', or 'hierarchical'
            n_clusters: Number of clusters (not used for DBSCAN)
            
        Returns:
            Cluster labels and fitted model
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
            
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        elif method == 'dbscan':
            from sklearn.cluster import DBSCAN
            model = DBSCAN(eps=0.5, min_samples=5)
        elif method == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
            
        labels = model.fit_predict(X)
        return labels, model
    
    def plot_clusters(self, X, labels, title='Clustering Results'):
        """
        Plot clustering results (works for 2D data)
        
        Args:
            X: Input data (first two dimensions will be used for plotting)
            labels: Cluster labels
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()


def get_available_models(task='regression'):
    """
    Get a dictionary of available models for a given task
    
    Args:
        task: 'regression' or 'classification'
        
    Returns:
        Dictionary of model names and instances
    """
    if task == 'regression':
        return {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
            'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
            'CatBoost': CatBoostRegressor(random_state=42, verbose=0)
        }
    else:  # classification
        return {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
            'Ridge Classifier': RidgeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, verbosity=0),
            'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=0)
        }


def create_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Create a preprocessing pipeline for numeric and categorical features
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        
    Returns:
        Preprocessing pipeline
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor


def hyperparameter_tuning(model, param_grid, X, y, cv=5, scoring=None, search_type='grid'):
    """
    Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
    
    Args:
        model: Model instance
        param_grid: Dictionary of parameters to search
        X: Features
        y: Target
        cv: Number of cross-validation folds
        scoring: Scoring metric
        search_type: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
        
    Returns:
        Best model and results
    """
    if search_type == 'grid':
        search = GridSearchCV(
            model, param_grid, 
            cv=cv, 
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
    else:  # random search
        search = RandomizedSearchCV(
            model, param_grid, 
            cv=cv, 
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            n_iter=10,  # Number of parameter settings sampled
            random_state=42
        )
    
    search.fit(X, y)
    
    print(f"Best parameters: {search.best_params_}")
    print(f"Best {scoring if scoring else 'score'}: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.cv_results_
