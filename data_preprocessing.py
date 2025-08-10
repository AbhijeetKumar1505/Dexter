"""
Enhanced Data Preprocessing Module

This module provides comprehensive data preprocessing capabilities including:
- Advanced data cleaning
- Text preprocessing
- Feature engineering
- Automated data transformation pipelines
"""

import pandas as pd
import numpy as np
import re
import string
from typing import Union, List, Dict, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, LabelEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.compose import ColumnTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class DataCleaner(BaseEstimator, TransformerMixin):
    """Advanced data cleaning operations"""
    
    def __init__(self, 
                 drop_duplicates: bool = True,
                 handle_missing: str = 'auto',  # 'drop', 'mean', 'median', 'mode', 'knn', 'ffill', 'bfill'
                 outlier_method: str = None,    # 'zscore', 'iqr', 'isolation_forest'
                 text_columns: List[str] = None,
                 datetime_columns: List[str] = None):
        self.drop_duplicates = drop_duplicates
        self.handle_missing = handle_missing
        self.outlier_method = outlier_method
        self.text_columns = text_columns or []
        self.datetime_columns = datetime_columns or []
        self.imputers_ = {}
        self.scaler_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        # Handle datetime columns
        for col in self.datetime_columns:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], errors='coerce')
                
        # Store column dtypes for inverse transform
        self.dtypes_ = X.dtypes
        
        # Handle missing values
        if self.handle_missing in ['mean', 'median', 'most_frequent']:
            self.imputer_ = SimpleImputer(strategy=self.handle_missing)
            numeric_cols = X.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                self.imputer_.fit(X[numeric_cols])
        elif self.handle_missing == 'knn':
            self.imputer_ = KNNImputer(n_neighbors=5)
            numeric_cols = X.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                self.imputer_.fit(X[numeric_cols])
                
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Drop duplicates
        if self.drop_duplicates:
            X = X.drop_duplicates()
            
        # Handle datetime columns
        for col in self.datetime_columns:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], errors='coerce')
                # Extract datetime features
                X[f'{col}_year'] = X[col].dt.year
                X[f'{col}_month'] = X[col].dt.month
                X[f'{col}_day'] = X[col].dt.day
                X[f'{col}_hour'] = X[col].dt.hour
                X[f'{col}_dayofweek'] = X[col].dt.dayofweek
                X = X.drop(columns=[col])
        
        # Handle missing values
        if hasattr(self, 'imputer_'):
            numeric_cols = X.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                X[numeric_cols] = self.imputer_.transform(X[numeric_cols])
        
        # Handle outliers
        if self.outlier_method == 'zscore':
            numeric_cols = X.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
                X = X[z_scores < 3]  # Remove rows with z-score > 3
                
        return X


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Advanced text preprocessing"""
    
    def __init__(self, 
                 text_columns: List[str] = None,
                 remove_punctuation: bool = True,
                 to_lowercase: bool = True,
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 remove_numbers: bool = True,
                 remove_special_chars: bool = True):
        self.text_columns = text_columns or []
        self.remove_punctuation = remove_punctuation
        self.to_lowercase = to_lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_numbers = remove_numbers
        self.remove_special_chars = remove_special_chars
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        for col in self.text_columns:
            if col in X.columns:
                X[col] = X[col].astype(str).apply(self._preprocess_text)
                
        return X
    
    def _preprocess_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
            
        if self.to_lowercase:
            text = text.lower()
            
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
            
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
            
        if self.remove_special_chars:
            text = re.sub(r'[^\w\s]', '', text)
            
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
            
        # Lemmatization
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            
        return ' '.join(tokens)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering"""
    
    def __init__(self, 
                 create_interactions: bool = True,
                 polynomial_degree: int = 2,
                 create_statistical_features: bool = True,
                 numeric_columns: List[str] = None,
                 categorical_columns: List[str] = None):
        self.create_interactions = create_interactions
        self.polynomial_degree = polynomial_degree
        self.create_statistical_features = create_statistical_features
        self.numeric_columns = numeric_columns or []
        self.categorical_columns = categorical_columns or []
        self.poly_features_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Create interaction features
        if self.create_interactions and len(self.numeric_columns) > 1:
            for i in range(len(self.numeric_columns)):
                for j in range(i + 1, len(self.numeric_columns)):
                    col1, col2 = self.numeric_columns[i], self.numeric_columns[j]
                    X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
        
        # Create polynomial features
        if self.polynomial_degree > 1 and self.numeric_columns:
            for col in self.numeric_columns:
                for degree in range(2, self.polynomial_degree + 1):
                    X[f'{col}_pow_{degree}'] = X[col] ** degree
        
        # Create statistical features
        if self.create_statistical_features and len(self.numeric_columns) > 1:
            X['mean'] = X[self.numeric_columns].mean(axis=1)
            X['std'] = X[self.numeric_columns].std(axis=1)
            X['min'] = X[self.numeric_columns].min(axis=1)
            X['max'] = X[self.numeric_columns].max(axis=1)
            X['range'] = X['max'] - X['min']
        
        return X


def create_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str] = None,
    text_features: List[str] = None,
    datetime_features: List[str] = None,
    feature_engineering_params: Dict = None
) -> ColumnTransformer:
    """
    Create a comprehensive preprocessing pipeline
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        text_features: List of text feature names
        datetime_features: List of datetime feature names
        feature_engineering_params: Parameters for feature engineering
        
    Returns:
        A configured ColumnTransformer with all preprocessing steps
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    text_transformer = Pipeline(steps=[
        ('preprocessor', TextPreprocessor(text_columns=text_features)),
        ('vectorizer', TfidfVectorizer(max_features=100))
    ])
    
    # Create feature engineering steps
    feature_engineering_params = feature_engineering_params or {}
    feature_engineering = FeatureEngineer(
        create_interactions=feature_engineering_params.get('create_interactions', True),
        polynomial_degree=feature_engineering_params.get('polynomial_degree', 2),
        create_statistical_features=feature_engineering_params.get('create_statistical_features', True),
        numeric_columns=numeric_features,
        categorical_columns=categorical_features or []
    )
    
    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features) if numeric_features else [],
            ('cat', categorical_transformer, categorical_features) if categorical_features else [],
            ('text', text_transformer, text_features) if text_features else []
        ],
        remainder='drop'
    )
    
    # Create final pipeline with feature engineering
    pipeline = Pipeline([
        ('cleaner', DataCleaner(datetime_columns=datetime_features or [])),
        ('feature_engineering', feature_engineering),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline


def get_feature_names(column_transformer):
    """Get feature names from a ColumnTransformer"""
    feature_names = []
    
    for name, transformer, columns in column_transformer.transformers_:
        if transformer == 'drop':
            continue
            
        if hasattr(transformer, 'get_feature_names_out'):
            names = transformer.get_feature_names_out(columns)
        else:
            names = columns
            
        feature_names.extend(names)
        
    return feature_names
