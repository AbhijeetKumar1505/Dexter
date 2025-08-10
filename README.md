# 🚀 Data Intelligence Dynamo (DID)

Welcome to Data Intelligence Dynamo (DID) - Where Data Meets AI in Perfect Harmony! 

DID is not just another data analysis tool; it's your personal data science team in a box. Built with Streamlit and powered by Together AI, DID transforms raw data into actionable insights through an intuitive, conversation-driven interface. Whether you're a data scientist, business analyst, or just data-curious, DID makes advanced analytics accessible to everyone.

## ✨ Key Features

### 🎯 Data Analysis Pipeline: Your Data's Journey to Insights

Imagine your data going on a magical journey through our intelligent pipeline:

1. **Data Ingestion Station**
   - Drag-and-drop your CSV/Excel files
   - Instant data preview to ensure everything looks shipshape
   - Smart type detection for each column

2. **Data Wrangling Workshop**
   - Interactive data cleaning with a single click
   - Missing value imputation with intelligent defaults
   - Outlier detection and treatment suggestions

3. **Exploration Playground**
   - One-click visualizations that adapt to your data
   - Statistical insights that actually make sense
   - Interactive plots that tell the story behind your numbers

4. **AI-Powered Analysis**
   - Natural language queries about your data
   - Automated insights and pattern detection
   - Smart visualization recommendations

5. **Model Training Arena**
   - From simple regressions to complex neural networks
   - Automated model selection and hyperparameter tuning
   - Performance metrics that matter

6. **Deployment Launchpad**
   - One-click model deployment
   - API endpoint generation
   - Shareable dashboards

### 🛠️ Core Features

- 📊 Data visualization (scatter plots, bar charts, histograms, box plots, violin plots)
- 📈 Statistical analysis (hypothesis testing, correlation analysis)
- 🤖 Machine learning models (Linear Regression, Decision Trees, Random Forest)
- 🧠 TensorFlow hybrid neural network models for advanced analysis
- 💡 AI-powered data insights 
- 📝 Interactive data preprocessing options
- 📊 Summary statistics and data exploration tools

## Advanced ML Models

The dashboard now includes TensorFlow-based hybrid models for more accurate analysis:

- **Dense Neural Networks**: Standard fully-connected neural networks
- **Wide & Deep Networks**: Google-inspired architecture that combines memorization and generalization
- **ResNet Architecture**: Uses residual connections for better gradient flow in deep networks
- **Transformer-based Models**: Leverages self-attention mechanisms for improved performance on tabular data

These advanced models typically outperform traditional ML approaches on complex datasets.

## Setup

1. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Secrets:
Create a `.streamlit/secrets.toml` file in your project directory with:
```toml
TOGETHER_API_KEY = "your-together-ai-api-key"
```

## Running the Application

1. Ensure your virtual environment is activated
2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Upload your CSV file using the sidebar
2. Choose from various data preprocessing options
3. Explore visualizations and statistical analyses
4. Train and evaluate machine learning models
5. Use TensorFlow hybrid models for more accurate predictions
6. Ask questions about your data using natural language

## Requirements

- Python 3.8+
- TensorFlow 2.15+
- See requirements.txt for full list of dependencies

## Getting an API Key

1. Visit [Together AI](https://together.ai)
2. Create an account
3. Navigate to the API section
4. Copy your API key
5. Add it to `.streamlit/secrets.toml`

## Contributing

Feel free to open issues or submit pull requests for any improvements.

## License

MIT License 