# Interactive Data Analysis Dashboard

An interactive data analysis dashboard built with Streamlit that allows users to upload CSV files, perform data analysis, create visualizations, train machine learning models, and ask questions about their data using DeepSeek AI.

## Features

- 📊 Data visualization (scatter plots, bar charts, histograms, box plots, violin plots)
- 📈 Statistical analysis (hypothesis testing, correlation analysis)
- 🤖 Machine learning models (Linear Regression, Decision Trees, Random Forest)
- 💡 AI-powered data insights using DeepSeek
- 📝 Interactive data preprocessing options
- 📊 Summary statistics and data exploration tools

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
5. Ask questions about your data using natural language

## Requirements

- Python 3.8+
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