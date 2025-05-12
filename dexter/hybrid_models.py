import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Tuple, Dict, List, Union, Optional
import matplotlib.pyplot as plt

class HybridModel:
    """
    Base class for TensorFlow hybrid models
    """
    def __init__(self, name: str = "hybrid_model"):
        self.name = name
        self.model = None
        self.history = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = None  # Will be set during preprocessing
        
    def preprocess(self, X: pd.DataFrame, y: pd.Series, scale_target: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for TensorFlow models
        
        Args:
            X: Features DataFrame
            y: Target Series
            scale_target: Whether to scale the target variable
            
        Returns:
            Tuple of preprocessed X and y as numpy arrays
        """
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Optionally scale target
        if scale_target:
            self.target_scaler = MinMaxScaler()
            y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        else:
            y_scaled = y.values
            
        return X_scaled, y_scaled
    
    def build_model(self, input_shape: Tuple[int]) -> None:
        """
        Build the model architecture
        
        Args:
            input_shape: Shape of input features
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              epochs: int = 50, batch_size: int = 32, verbose: int = 1) -> Dict:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            validation_data: Tuple of (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1],))
            
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        return self.history.history
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features DataFrame or array
            
        Returns:
            Array of predictions
        """
        # Check if X is a DataFrame and convert if needed
        if isinstance(X, pd.DataFrame):
            X = self.feature_scaler.transform(X)
            
        # Make predictions
        preds = self.model.predict(X)
        
        # Inverse transform if target was scaled
        if self.target_scaler is not None:
            if len(preds.shape) == 1:
                preds = preds.reshape(-1, 1)
            preds = self.target_scaler.inverse_transform(preds).flatten()
            
        return preds
    
    def evaluate(self, X_test: Union[pd.DataFrame, np.ndarray], 
                y_test: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert to numpy arrays if needed
        if isinstance(X_test, pd.DataFrame):
            X_test_np = self.feature_scaler.transform(X_test)
        else:
            X_test_np = X_test
            
        if isinstance(y_test, pd.Series):
            y_test_np = y_test.values
        else:
            y_test_np = y_test
            
        # Make predictions
        y_pred = self.predict(X_test_np)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_np, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_np, y_pred)
        r2 = r2_score(y_test_np, y_pred)
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
    
    def plot_history(self) -> plt.Figure:
        """
        Plot training history
        
        Returns:
            Matplotlib figure
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet")
            
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax[0].plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            ax[0].plot(self.history.history['val_loss'], label='Validation Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Training and Validation Loss')
        ax[0].legend()
        
        # Plot MAE or other metrics if available
        metric_keys = [k for k in self.history.history.keys() 
                     if k not in ['loss', 'val_loss'] and not k.startswith('val_')]
        
        if metric_keys:
            metric = metric_keys[0]
            ax[1].plot(self.history.history[metric], label=f'Train {metric}')
            val_metric = f'val_{metric}'
            if val_metric in self.history.history:
                ax[1].plot(self.history.history[val_metric], label=f'Validation {metric}')
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel(metric)
            ax[1].set_title(f'Training and Validation {metric}')
            ax[1].legend()
        
        plt.tight_layout()
        return fig


class DenseNNRegressor(HybridModel):
    """
    Dense Neural Network Regressor model
    """
    def __init__(self, name: str = "dense_nn", hidden_layers: List[int] = [64, 32]):
        super().__init__(name=name)
        self.hidden_layers = hidden_layers
    
    def build_model(self, input_shape: Tuple[int]) -> None:
        """
        Build a dense neural network model
        
        Args:
            input_shape: Shape of input features
        """
        model = keras.Sequential(name=self.name)
        
        # Input layer
        model.add(keras.layers.Input(shape=input_shape))
        
        # Hidden layers
        for units in self.hidden_layers:
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(0.2))
            
        # Output layer
        model.add(keras.layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model


class WideAndDeepModel(HybridModel):
    """
    Wide and Deep hybrid model for tabular data
    """
    def __init__(self, name: str = "wide_and_deep", 
                 hidden_layers: List[int] = [64, 32],
                 l2_regularization: float = 0.01):
        super().__init__(name=name)
        self.hidden_layers = hidden_layers
        self.l2_regularization = l2_regularization
    
    def build_model(self, input_shape: Tuple[int]) -> None:
        """
        Build a Wide & Deep model
        
        Args:
            input_shape: Shape of input features
        """
        # Input layer
        inputs = keras.layers.Input(shape=input_shape)
        
        # Deep branch
        deep = keras.layers.Dense(self.hidden_layers[0], activation='relu')(inputs)
        deep = keras.layers.BatchNormalization()(deep)
        deep = keras.layers.Dropout(0.2)(deep)
        
        for units in self.hidden_layers[1:]:
            deep = keras.layers.Dense(units, activation='relu')(deep)
            deep = keras.layers.BatchNormalization()(deep)
            deep = keras.layers.Dropout(0.2)(deep)
        
        # Wide branch
        wide = keras.layers.Dense(
            units=16,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_regularization)
        )(inputs)
        
        # Combine wide and deep
        combined = keras.layers.concatenate([wide, deep])
        
        # Output
        output = keras.layers.Dense(1)(combined)
        
        # Create and compile model
        model = keras.Model(inputs=inputs, outputs=output, name=self.name)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model


class ResidualBlock(keras.layers.Layer):
    """Residual block for ResNet-style architecture"""
    
    def __init__(self, units: int, dropout_rate: float = 0.2):
        super().__init__()
        self.dense1 = keras.layers.Dense(units, activation='relu')
        self.batchnorm1 = keras.layers.BatchNormalization()
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dense2 = keras.layers.Dense(units)
        self.batchnorm2 = keras.layers.BatchNormalization()
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.add = keras.layers.Add()
        self.activation = keras.layers.Activation('relu')
        
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.batchnorm1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.batchnorm2(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.add([x, inputs])
        return self.activation(x)


class ResNetRegressor(HybridModel):
    """
    ResNet-style regressor for tabular data
    """
    def __init__(self, name: str = "resnet_regressor", blocks: int = 2, units: int = 64):
        super().__init__(name=name)
        self.blocks = blocks
        self.units = units
    
    def build_model(self, input_shape: Tuple[int]) -> None:
        """
        Build a ResNet-style model
        
        Args:
            input_shape: Shape of input features
        """
        inputs = keras.layers.Input(shape=input_shape)
        
        # Initial mapping to correct dimensions
        x = keras.layers.Dense(self.units, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        
        # ResNet blocks
        for _ in range(self.blocks):
            x = ResidualBlock(self.units)(x)
        
        # Output layer
        outputs = keras.layers.Dense(1)(x)
        
        # Create and compile model
        model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model


class TransformerRegressor(HybridModel):
    """
    Transformer-based regressor for tabular data
    """
    def __init__(self, name: str = "transformer_regressor", 
                 head_size: int = 128, num_heads: int = 2, 
                 ff_dim: int = 128, dropout: float = 0.2):
        super().__init__(name=name)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
    
    def build_model(self, input_shape: Tuple[int]) -> None:
        """
        Build a Transformer-based model
        
        Args:
            input_shape: Shape of input features
        """
        inputs = keras.layers.Input(shape=input_shape)
        
        # Feature embedding
        x = keras.layers.Dense(self.head_size, activation='relu')(inputs)
        x = keras.layers.Reshape((1, self.head_size))(x)  # Add sequence dimension
        
        # Self-attention
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_size // self.num_heads
        )(x, x)
        
        # Skip connection 1
        x = keras.layers.Add()([attention_output, x])
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn_output = keras.Sequential([
            keras.layers.Dense(self.ff_dim, activation='relu'),
            keras.layers.Dense(self.head_size),
            keras.layers.Dropout(self.dropout)
        ])(x)
        
        # Skip connection 2
        x = keras.layers.Add()([ffn_output, x])
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Flatten and output
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1)(x)
        
        # Create and compile model
        model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model


def get_hybrid_model(model_type: str, **kwargs) -> HybridModel:
    """
    Factory function to get a hybrid model
    
    Args:
        model_type: Type of model to create
        **kwargs: Additional arguments for the model
        
    Returns:
        Instantiated model
    """
    model_types = {
        'dense': DenseNNRegressor,
        'wide_and_deep': WideAndDeepModel,
        'resnet': ResNetRegressor,
        'transformer': TransformerRegressor
    }
    
    if model_type not in model_types:
        raise ValueError(f"Model type '{model_type}' not recognized. Available models: {list(model_types.keys())}")
    
    return model_types[model_type](**kwargs)


def prepare_data_for_hybrid_models(df: pd.DataFrame, target_column: str, test_size: float = 0.2, 
                                 val_size: float = 0.1, random_state: int = 42) -> Dict:
    """
    Prepare data for hybrid models
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed
        
    Returns:
        Dictionary with prepared data splits
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # First split: training+validation and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: training and validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


def evaluate_and_compare_models(models: List[HybridModel], data: Dict, baseline_models: Dict = None) -> Dict:
    """
    Evaluate and compare multiple hybrid models
    
    Args:
        models: List of HybridModel instances
        data: Dictionary with data splits
        baseline_models: Optional dictionary of baseline models
        
    Returns:
        Dictionary with evaluation results
    """
    results = {}
    
    # Evaluate hybrid models
    for model in models:
        model_results = model.evaluate(data['X_test'], data['y_test'])
        results[model.name] = model_results
    
    # Evaluate baseline models if provided
    if baseline_models:
        for name, model in baseline_models.items():
            # Sklearn-style model
            y_pred = model.predict(data['X_test'])
            mse = mean_squared_error(data['y_test'], y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(data['y_test'], y_pred)
            r2 = r2_score(data['y_test'], y_pred)
            
            results[name] = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
    
    return results


def plot_model_comparison(results: Dict, metric: str = 'rmse') -> plt.Figure:
    """
    Plot model comparison
    
    Args:
        results: Dictionary with evaluation results
        metric: Metric to compare
        
    Returns:
        Matplotlib figure
    """
    models = list(results.keys())
    values = [results[model][metric] for model in models]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar chart
    ax.bar(models, values)
    ax.set_ylabel(metric.upper())
    ax.set_title(f'Model Comparison - {metric.upper()}')
    
    # Add values on top of bars
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig 