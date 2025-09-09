# AI-Powered Automated Data Analysis & PDF Report Generator

An intelligent Streamlit application that performs automated exploratory data analysis (EDA) on CSV and Excel files, generates insightful visualizations, and creates comprehensive PDF reports with AI-powered insights.

---


## Features

### Smart Data Processing
- **Intelligent Column Detection**: Automatically identifies and excludes identifier columns (IDs, names, etc.) based on uniqueness ratio
- **Automatic Data Cleaning**: Handles missing values, removes constant columns, and filters out low-quality rows
- **Data Type Recognition**: Automatically detects and converts datetime columns
- **Advanced Filtering**: Interactive date range filtering for temporal data

### Comprehensive Visualizations
- **Numerical Analysis**: Histograms with statistical insights
- **Time Series Analysis**: Interactive time-based plots
- **Categorical Analysis**: Bar charts for categorical data distribution
- **Multi-variate Analysis**: 
  - Box plots and violin plots (numerical vs categorical)
  - Scatter plots with optional regression lines (numerical vs numerical)
  - Stacked bar charts and heatmaps (categorical vs categorical)

### AI-Powered Insights
- **Automated Chart Analysis**: AI-generated insights for each visualization
- **Interactive Q&A**: Ask questions about your data and get GPT-powered answers
- **Smart Column Filtering**: Automatically excludes non-meaningful columns from analysis

### Professional PDF Reports
- **Custom Report Generation**: Select specific visualizations for inclusion
- **Comprehensive Documentation**: Includes all insights and analysis results
- **Downloadable Format**: Professional PDF reports ready for sharing

---


## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/yilgara/ai-powered-data-analyst.git
   cd ai-powered-data-analyst
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API key**
   Create the file called ```secrets.toml``` in the ```.streamlit``` folder (or use Streamlit Cloud secrets)
   ```bash
   OPENAI_API_KEY = "your_openai_api_key_here"
   ```


5. **Run the application**
   ```bash
   streamlit run main.py
   ```

---

## Usage

### 1. **Upload Your Data**
- Supported formats: CSV, Excel (.xlsx, .xls)
- The tool automatically detects column types and data structure

### 2. **Data Processing**
- View filtered, and cleaned versions of your data
- The system automatically excludes identifier columns (IDs, names) that aren't suitable for visualization
- Apply date filters if your data contains temporal information

### 3. **Explore Visualizations**
- **Numerical Analysis**: Generate histograms for numerical columns
- **Time Series**: Analyze trends over time
- **Categorical Analysis**: Understand category distributions
- **Relationship Analysis**: Explore relationships between different variables

### 4. **Generate Reports**
- Select specific visualizations to include in your PDF report
- Add a custom title for your report
- Download a professional PDF containing all selected analyses

### 5. **Ask Questions**
- Use the AI-powered Q&A feature to get specific insights about your data
- Ask multiple questions and get comprehensive answers

---

## Project Structure

```
automated-data-analysis/
├── main.py                 # Main Streamlit application
├── utils/
│   ├── chart.py           # Chart generation functions
│   ├── prompt.py          # AI prompt handling
│   └── pdf.py             # PDF report generation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

## Deployment

The application is deployed and accessible at:  
**[https://ai-powered-data-analyst-migdjvngemtdcigeepqrxb.streamlit.app](https://ai-powered-data-analyst-migdjvngemtdcigeepqrxb.streamlit.app)**


