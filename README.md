# Data Cleaning Tool

A Python-based data cleaning tool built with Streamlit that allows users to easily clean and prepare their data for analysis.

## Features

- **Data Upload**: Support for CSV and Excel files
- **Data Exploration**: View statistics, preview data, and visualize distributions
- **Data Cleaning Operations**:
  - Remove duplicate rows
  - Handle missing values (mean, median, mode, custom values, etc.)
  - Remove outliers (IQR or Z-Score methods)
  - Standardize or normalize numeric columns
  - Rename or drop columns
- **Cleaning History**: Track all cleaning operations applied to your data
- **Export**: Download the cleaned data as a CSV file

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone this repository or download the source code

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`

3. Use the sidebar to upload your data file (CSV or Excel)

4. Explore your data with the provided visualizations and statistics

5. Apply cleaning operations as needed:
   - Select an operation from the dropdown menu in the sidebar
   - Configure the operation parameters
   - Click the button to apply the operation

6. When finished, download your cleaned data using the download button

## Example Workflow

1. Upload a CSV file with customer data
2. Identify columns with missing values
3. Fill missing values using appropriate methods (e.g., mean for numeric columns)
4. Remove outliers from numeric columns
5. Standardize numeric features for machine learning
6. Rename columns to more descriptive names
7. Download the cleaned dataset

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 