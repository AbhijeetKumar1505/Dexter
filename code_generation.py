"""
Code Generation Module

This module provides functionality for generating, exporting, and documenting
Python code for data analysis and machine learning workflows.
"""
import os
import inspect
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import pandas as pd
import streamlit as st
from pathlib import Path

class CodeGenerator:
    """
    A class to handle generation and export of analysis and ML workflow code.
    """
    
    def __init__(self, df: pd.DataFrame = None):
        """
        Initialize the CodeGenerator with optional dataframe.
        
        Args:
            df: Optional pandas DataFrame for code generation context
        """
        self.df = df
        self.code_snippets = {}
        self.imports = set()
        self._setup_default_imports()
    
    def _setup_default_imports(self):
        """Set up default imports for generated code."""
        self.imports.update([
            'import pandas as pd',
            'import numpy as np',
            'import matplotlib.pyplot as plt',
            'import seaborn as sns',
            'from sklearn.model_selection import train_test_split',
            'from sklearn.preprocessing import StandardScaler, OneHotEncoder',
            'from sklearn.pipeline import Pipeline',
            'from sklearn.compose import ColumnTransformer',
            'from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score',
        ])
    
    def add_import(self, module: str, imports: List[str] = None, alias: str = None):
        """
        Add import statements to the generated code.
        
        Args:
            module: Module to import from
            imports: List of items to import (if None, imports the entire module)
            alias: Optional alias for the import
        """
        if imports:
            # Join the imports first, then use in the f-string
            imports_str = ', '.join(imports)
            import_str = f'from {module} import {imports_str}'
        else:
            import_str = f'import {module}'
            if alias:
                import_str += f' as {alias}'
        self.imports.add(import_str)
    
    def generate_loading_code(self, filepath: str = None) -> str:
        """
        Generate code for loading data.
        
        Args:
            filepath: Path to the data file (optional)
            
        Returns:
            str: Generated code for loading data
        """
        if filepath:
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.csv':
                return f"df = pd.read_csv('{filepath}')\n"
            elif ext in ['.xls', '.xlsx']:
                return f"df = pd.read_excel('{filepath}')\n"
            elif ext == '.json':
                return f"df = pd.read_json('{filepath}')\n"
        return "# Load your data here\ndf = pd.DataFrame()\n"
    
    def generate_cleaning_code(self, cleaning_steps: Dict[str, Any]) -> str:
        """
        Generate code for data cleaning steps.
        
        Args:
            cleaning_steps: Dictionary containing cleaning steps and parameters
            
        Returns:
            str: Generated code for data cleaning
        """
        code = "# Data Cleaning\n"
        
        if not cleaning_steps:
            return code + "# No cleaning steps specified\n"
        
        for step, params in cleaning_steps.items():
            if step == 'handle_missing':
                if params['method'] == 'drop':
                    code += "# Drop rows with missing values\n"
                    code += "df = df.dropna()\n\n"
                elif params['method'] == 'fill':
                    for col, val in params['values'].items():
                        code += f"# Fill missing values in {col}\n"
                        if isinstance(val, str):
                            code += f"df['{col}'] = df['{col}'].fillna('{val}')\n\n"
                        else:
                            code += f"df['{col}'] = df['{col}'].fillna({val})\n\n"
            elif step == 'drop_duplicates':
                code += "# Drop duplicate rows\n"
                code += "df = df.drop_duplicates()\n\n"
            elif step == 'convert_types':
                for col, dtype in params.items():
                    code += f"# Convert {col} to {dtype}\n"
                    code += f"df['{col}'] = df['{col}'].astype('{dtype}')\n\n"
        
        return code
    
    def generate_eda_code(self, eda_steps: Dict[str, Any]) -> str:
        """
        Generate code for exploratory data analysis.
        
        Args:
            eda_steps: Dictionary containing EDA steps and parameters
            
        Returns:
            str: Generated code for EDA
        """
        code = "# Exploratory Data Analysis\n"
        
        if not eda_steps:
            return code + "# No EDA steps specified\n"
        
        if eda_steps.get('show_info', False):
            code += "# Display DataFrame info\n"
            code += "print('\\n=== DataFrame Info ===')\n"
            code += "df.info()\n\n"
        
        if eda_steps.get('show_describe', False):
            code += "# Display descriptive statistics\n"
            code += "print('\\n=== Descriptive Statistics ===')\n"
            code += "print(df.describe())\n\n"
        
        if eda_steps.get('show_missing', False):
            code += "# Display missing values\n"
            code += "print('\\n=== Missing Values ===')\n"
            code += "print(df.isnull().sum())\n\n"
        
        if eda_steps.get('show_correlation', False):
            code += "# Calculate and display correlation matrix\n"
            code += "numeric_cols = df.select_dtypes(include=['number']).columns\n"
            code += "if len(numeric_cols) > 1:\n"
            code += "    print('\\n=== Correlation Matrix ===')\n"
            code += "    corr = df[numeric_cols].corr()\n"
            code += "    print(corr)\n\n"
            code += "    # Plot correlation heatmap\n"
            code += "    plt.figure(figsize=(10, 8))\n"
            code += "    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)\n"
            code += "    plt.title('Correlation Heatmap')\n"
            code += "    plt.tight_layout()\n"
            code += "    plt.show()\n\n"
        
        return code
    
    def generate_feature_engineering_code(self, feature_steps: Dict[str, Any]) -> str:
        """
        Generate code for feature engineering.
        
        Args:
            feature_steps: Dictionary containing feature engineering steps and parameters
            
        Returns:
            str: Generated code for feature engineering
        """
        code = "# Feature Engineering\n"
        
        if not feature_steps:
            return code + "# No feature engineering steps specified\n"
        
        if feature_steps.get('create_interactions', False):
            code += "# Create interaction features\n"
            code += "# Example: df['feature1_x_feature2'] = df['feature1'] * df['feature2']\n\n"
        
        if feature_steps.get('create_polynomials', False):
            degree = feature_steps.get('polynomial_degree', 2)
            code += f"# Create polynomial features (degree {degree})\n"
            code += "from sklearn.preprocessing import PolynomialFeatures\n"
            code += f"poly = PolynomialFeatures(degree={degree}, include_bias=False)\n"
            code += "# Select numeric columns for polynomial features\n"
            code += "numeric_cols = df.select_dtypes(include=['number']).columns\n"
            code += "if len(numeric_cols) > 0:\n"
            code += "    poly_features = poly.fit_transform(df[numeric_cols])\n"
            code += f"    poly_cols = [f'poly_{{i+1}}' for i in range(poly_features.shape[1])]\n"
            code += "    df_poly = pd.DataFrame(poly_features, columns=poly_cols)\n"
            code += "    df = pd.concat([df, df_poly], axis=1)\n\n"
        
        if feature_steps.get('create_stats', False):
            code += "# Create statistical features\n"
            code += "# Example: df['feature1_rolling_mean'] = df['feature1'].rolling(window=3).mean()\n\n"
        
        return code
    
    def generate_ml_code(
        self, 
        model_type: str, 
        target_col: str,
        feature_cols: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
        hyperparams: Dict[str, Any] = None
    ) -> str:
        """
        Generate code for training a machine learning model.
        
        Args:
            model_type: Type of model ('regression' or 'classification')
            target_col: Name of the target column
            feature_cols: List of feature columns
            test_size: Size of the test set (default: 0.2)
            random_state: Random seed (default: 42)
            hyperparams: Dictionary of hyperparameters for the model
            
        Returns:
            str: Generated code for training the model
        """
        code = f"# Machine Learning: {model_type.capitalize()}\n"
        # Split data
        code += "# Split data into features and target\n"
        code += f"X = df[{feature_cols}]\n"
        code += f"y = df['{target_col}']\n\n"
        
        code += "# Split data into training and testing sets\n"
        code += f"X_train, X_test, y_train, y_test = train_test_split(\n"
        code += f"    X, y, test_size={test_size}, random_state={random_state}\n)\n\n"
        
        # Preprocessing
        code += "# Preprocessing pipeline\n"
        code += "numeric_features = X.select_dtypes(include=['number']).columns.tolist()\n"
        code += "categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()\n\n"
        
        code += "numeric_transformer = Pipeline(steps=[\n"
        code += "    ('imputer', SimpleImputer(strategy='median')),\n"
        code += "    ('scaler', StandardScaler())\n])\n\n"
        
        code += "categorical_transformer = Pipeline(steps=[\n"
        code += "    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\n"
        code += "    ('onehot', OneHotEncoder(handle_unknown='ignore'))\n])\n\n"
        
        code += "preprocessor = ColumnTransformer(\n    transformers=[\n        ('num', numeric_transformer, numeric_features),\n        ('cat', categorical_transformer, categorical_features)\n    ])\n\n"
        
        # Model training
        code += "# Create and train the model\n"
        
        if model_type.lower() == 'regression':
            code += self._generate_regression_code(hyperparams)
            metric_code = "r2 = r2_score(y_test, y_pred)\n"
            metric_code += "print(f'R² Score: {r2:.4f}')\n\n"
            metric_code += "mse = mean_squared_error(y_test, y_pred)\n"
            metric_code += "print(f'Mean Squared Error: {mse:.4f}')\n"
        else:  # classification
            code += self._generate_classification_code(hyperparams)
            metric_code = "print('\\n=== Classification Report ===')\n"
            metric_code += "print(classification_report(y_test, y_pred))\n\n"
            metric_code += "accuracy = accuracy_score(y_test, y_pred)\n"
            metric_code += "print(f'Accuracy: {accuracy:.4f}')\n"
        
        # Model evaluation
        code += "\n# Evaluate the model\n"
        code += "y_pred = model.predict(X_test)\n\n"
        code += metric_code
        
        return code
    
    def _generate_regression_code(self, hyperparams: Dict[str, Any] = None) -> str:
        """Generate code for regression model."""
        self.imports.add('from sklearn.ensemble import RandomForestRegressor')
        
        code = "# Initialize the model\n"
        code += "model = Pipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('regressor', RandomForestRegressor(\n"
        
        # Add hyperparameters if provided
        if hyperparams:
            for param, value in hyperparams.items():
                code += f"        {param}={value},\n"
        code += "    ))\n])\n\n"
        
        code += "# Train the model\n"
        code += "model.fit(X_train, y_train)\n\n"
        
        return code
    
    def _generate_classification_code(self, hyperparams: Dict[str, Any] = None) -> str:
        """Generate code for classification model."""
        self.imports.add('from sklearn.ensemble import RandomForestClassifier')
        
        code = "# Initialize the model\n"
        code += "model = Pipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('classifier', RandomForestClassifier(\n"
        
        # Add hyperparameters if provided
        if hyperparams:
            for param, value in hyperparams.items():
                code += f"        {param}={value},\n"
        code += "    ))\n])\n\n"
        
        code += "# Train the model\n"
        code += "model.fit(X_train, y_train)\n\n"
        
        return code
    
    def generate_visualization_code(self, vis_params: Dict[str, Any]) -> str:
        """
        Generate code for data visualization.
        
        Args:
            vis_params: Dictionary containing visualization parameters
            
        Returns:
            str: Generated code for data visualization
        """
        code = "# Data Visualization\n"
        
        if not vis_params:
            return code + "# No visualization parameters specified\n"
        
        self.imports.add('import matplotlib.pyplot as plt')
        self.imports.add('import seaborn as sns')
        
        code += "# Set the style for the plots\n"
        code += "sns.set_style('whitegrid')\n\n"
        
        if vis_params.get('plot_type') == 'histogram':
            col = vis_params.get('column', '')
            code += f"# Histogram of {col}\n"
            code += f"plt.figure(figsize=(10, 6))\n"
            code += f"sns.histplot(data=df, x='{col}', kde=True)\n"
            code += f"plt.title('Distribution of {col}')\n"
            code += "plt.show()\n\n"
        
        elif vis_params.get('plot_type') == 'scatter':
            x_col = vis_params.get('x_column', '')
            y_col = vis_params.get('y_column', '')
            hue = vis_params.get('hue')
            
            code += f"# Scatter plot of {x_col} vs {y_col}\n"
            code += f"plt.figure(figsize=(10, 6))\n"
            
            if hue:
                code += f"sns.scatterplot(data=df, x='{x_col}', y='{y_col}', hue='{hue}')\n"
            else:
                code += f"sns.scatterplot(data=df, x='{x_col}', y='{y_col}')\n"
            
            code += f"plt.title('{y_col} vs {x_col}')\n"
            code += "plt.show()\n\n"
        
        elif vis_params.get('plot_type') == 'boxplot':
            col = vis_params.get('column', '')
            by = vis_params.get('by')
            
            code += f"# Box plot of {col}\n"
            code += f"plt.figure(figsize=(10, 6))\n"
            
            if by:
                code += f"sns.boxplot(data=df, x='{by}', y='{col}')\n"
            else:
                code += f"sns.boxplot(data=df, y='{col}')\n"
            
            code += f"plt.title('Box Plot of {col}')\n"
            code += "plt.xticks(rotation=45)\n"
            code += "plt.tight_layout()\n"
            code += "plt.show()\n\n"
        
        return code
    
    def export_code(
        self, 
        code: str, 
        output_dir: str = '.', 
        filename: str = None,
        format: str = 'py',
        include_docs: bool = True
    ) -> str:
        """
        Export the generated code to a file.
        
        Args:
            code: The code to export
            output_dir: Output directory (default: current directory)
            filename: Output filename (without extension)
            format: Output format ('py', 'ipynb', 'md')
            include_docs: Whether to include documentation in the exported code
            
        Returns:
            str: Path to the exported file
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'analysis_{timestamp}'
        
        filepath = os.path.join(output_dir, f'{filename}.{format}')
        
        if format == 'ipynb':
            self._export_to_notebook(code, filepath, include_docs)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                if include_docs:
                    f.write(self._generate_file_header())
                f.write(code)
        
        return filepath
    
    def _export_to_notebook(self, code: str, filepath: str, include_docs: bool = True):
        """Export code to Jupyter notebook format."""
        try:
            import nbformat as nbf
            
            # Create a new notebook
            nb = nbf.v4.new_notebook()
            
            # Add markdown cell with documentation if needed
            if include_docs:
                nb['cells'].append(nbf.v4.new_markdown_cell(self._generate_file_header()))
            
            # Split code into cells based on double newlines
            code_blocks = [block for block in code.split('\n\n') if block.strip()]
            
            # Add code cells
            for block in code_blocks:
                nb['cells'].append(nbf.v4.new_code_cell(block))
            
            # Write the notebook to a file
            with open(filepath, 'w', encoding='utf-8') as f:
                nbf.write(nb, f)
                
        except ImportError:
            raise ImportError("The 'nbformat' package is required to export to Jupyter notebook format. "
                            "Please install it using 'pip install nbformat'")
    
    def _generate_file_header(self) -> str:
        """Generate a file header with metadata and documentation."""
        header = """# Auto-generated Analysis Code
# ===========================

# This code was automatically generated by the Data Analysis Tool.
# Date: {date}
# Python Version: {python_version}
# Dependencies: {dependencies}

""".format(
            date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            python_version='.'.join(map(str, sys.version_info[:3])),
            dependencies=', '.join(sorted({
                'pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn'
            }))
        )
        
        return header
    
    def generate_documentation(self, code: str, format: str = 'md') -> str:
        """
        Generate documentation for the given code.
        
        Args:
            code: The code to document
            format: Output format ('md' for Markdown, 'html' for HTML)
            
        Returns:
            str: Generated documentation in the specified format
        """
        if format not in ['md', 'html']:
            raise ValueError("Unsupported documentation format. Supported formats: 'md', 'html'")
        
        # Initialize documentation with header
        doc = """# Analysis Documentation
=====================

## Code Overview
This document provides documentation for the auto-generated analysis code.

## Dependencies
```
"""
        
        # Add imports section
        doc += "\n".join(sorted(self.imports))
        doc += """
```

## Code Explanation

"""
        
        # Parse and document code blocks
        lines = code.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Section headers
            if line.startswith('#'):
                if ':' in line:
                    current_section = line.strip('# ').split(':')[0]
                    doc += f"\n### {current_section}\n"
                    doc += f"{line.strip('# ').strip()}\n\n"
                else:
                    doc += f"\n**{line.strip('# ').strip()}**\n\n"
            # Code blocks
            elif not line.isspace():
                doc += f"```python\n{line}\n```\n\n"
        
        # Add footer
        doc += """
## Notes
- This code was automatically generated and may require adjustments for your specific use case.
- Make sure all required dependencies are installed before running the code.
"""
        
        if format == 'html':
            try:
                import markdown
                from bs4 import BeautifulSoup
                
                # Convert markdown to HTML
                html = markdown.markdown(doc)
                
                # Create a basic HTML template
                template = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Analysis Documentation</title>
                    <style>
                        body {{ 
                            font-family: Arial, sans-serif; 
                            line-height: 1.6; 
                            max-width: 800px; 
                            margin: 0 auto; 
                            padding: 20px; 
                        }}
                        h1 {{ 
                            color: #2c3e50; 
                            border-bottom: 2px solid #3498db; 
                        }}
                        h2 {{ 
                            color: #2980b9; 
                        }}
                        h3 {{ 
                            color: #3498db; 
                        }}
                        pre {{ 
                            background-color: #f5f5f5; 
                            padding: 10px; 
                            border-radius: 5px; 
                            overflow-x: auto; 
                        }}
                        code {{ 
                            font-family: 'Courier New', monospace; 
                        }}
                    </style>
                </head>
                <body>
                    {content}
                </body>
                </html>
                """
                
                # Insert content into template
                soup = BeautifulSoup(html, 'html.parser')
                content = soup.prettify()
                html = template.format(content=content)
                
                return html
                
            except ImportError:
                raise ImportError(
                    "HTML documentation requires 'markdown' and 'beautifulsoup4' packages. "
                    "Install them with 'pip install markdown beautifulsoup4'"
                )
        
        return doc

def export_to_github(gist_content: str, filename: str, description: str = "", public: bool = True) -> str:
    """
    Export code to a GitHub Gist.
    
    Args:
        gist_content: Content to upload to the Gist
        filename: Name of the file in the Gist
        description: Description of the Gist
        public: Whether the Gist should be public
        
    Returns:
        str: URL of the created Gist
    """
    try:
        from github import Github
        from github import InputFileContent
        
        github_token = os.getenv('GITHUB_TOKEN')
        
        if not github_token:
            return "Error: GITHUB_TOKEN environment variable not set. Please set it to your GitHub personal access token."
        
        g = Github(github_token)
        
        # Create a new Gist
        gist = g.get_user().create_gist(
            public=public,
            files={filename: InputFileContent(gist_content)},
            description=description
        )
        
        return f"Gist created successfully! URL: {gist.html_url}"
    
    except ImportError:
        return "Error: PyGithub package is required. Install it with 'pip install PyGithub'"
    except Exception as e:
        return f"Error creating Gist: {str(e)}"

def export_to_gitlab(snippet_content: str, filename: str, title: str = "", description: str = "", visibility: str = "public") -> str:
    """
    Export code to a GitLab Snippet.
    
    Args:
        snippet_content: Content to upload to the Snippet
        filename: Name of the file in the Snippet
        title: Title of the Snippet
        description: Description of the Snippet
        visibility: Visibility of the Snippet ('private', 'internal', 'public')
        
    Returns:
        str: URL of the created Snippet or error message
    """
    try:
        import gitlab
        
        gitlab_token = os.getenv('GITLAB_TOKEN')
        gitlab_url = os.getenv('GITLAB_URL', 'https://gitlab.com')
        
        if not gitlab_token:
            return "Error: GITLAB_TOKEN environment variable not set. Please set it to your GitLab personal access token."
        
        gl = gitlab.Gitlab(url=gitlab_url, private_token=gitlab_token)
        
        # Create a new snippet
        snippet = gl.snippets.create({
            'title': title or f"Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            'file_name': filename,
            'content': snippet_content,
            'visibility': visibility
        })
        
        return f"Snippet created successfully! URL: {snippet.web_url}"
    
    except ImportError:
        return "Error: python-gitlab package is required. Install it with 'pip install python-gitlab'"
    except Exception as e:
        return f"Error creating GitLab snippet: {str(e)}"

def export_to_bitbucket(snippet_content: str, filename: str, title: str = "", description: str = "", is_private: bool = True) -> str:
    """
    Export code to a Bitbucket Snippet.
    
    Args:
        snippet_content: Content to upload to the Snippet
        filename: Name of the file in the Snippet
        title: Title of the Snippet
        description: Description of the Snippet
        is_private: Whether the Snippet should be private
        
    Returns:
        str: URL of the created Snippet or error message
    """
    try:
        from atlassian import Bitbucket
        
        bitbucket_username = os.getenv('BITBUCKET_USERNAME')
        bitbucket_app_password = os.getenv('BITBUCKET_APP_PASSWORD')
        
        if not bitbucket_username or not bitbucket_app_password:
            return "Error: BITBUCKET_USERNAME and BITBUCKET_APP_PASSWORD environment variables must be set."
        
        bitbucket = Bitbucket(
            url='https://api.bitbucket.org/',
            username=bitbucket_username,
            password=bitbucket_app_password
        )
        
        # Create a new snippet
        snippet = bitbucket.create_snippet(
            title=title or f"Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            files={filename: {'content': snippet_content}},
            is_private=is_private
        )
        
        return f"Snippet created successfully! URL: {snippet['links']['html']['href']}"
    
    except ImportError:
        return "Error: atlassian-python-api package is required. Install it with 'pip install atlassian-python-api'"
    except Exception as e:
        return f"Error creating Bitbucket snippet: {str(e)}"
