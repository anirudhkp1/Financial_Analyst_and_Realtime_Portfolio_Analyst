-----

# Financial Insights & Portfolio Analysis System

This project provides a comprehensive suite for financial analysis, split into two main components:

1.  A **Personal Spending Analyzer** that performs Exploratory Data Analysis (EDA) on a local SQL database of transactions, generates visualizations, and uploads them to AWS S3.
2.  A **Real-Time Portfolio Analyst** that uses AutoGen's Model Context Protocol (MCP) to fetch live market data from Alpha Vantage for a specific stock and generate a detailed report.

-----

## 🚀 Core Features

  * **Personal Spending Analysis:** Conducts deep EDA on transaction data from a SQLite database.
  * **Real-Time Portfolio Analysis:** Fetches live market data, news, earnings, and fundamentals using the Alpha Vantage API.
  * **Multi-Agent Architecture:** Uses AutoGen agents for task orchestration, including a two-agent MCP workflow (`data_collector` -\> `report_writer`).
  * **Natural Language Database Query:** Integrates a LangChain SQL Agent (`query_db`) to answer natural language questions about personal spending data.
  * **Cloud Integration:** Automatically uploads generated Plotly graphs (e.g., spending by category, daily trends) to an AWS S3 bucket.
  * **Modern AutoGen Tooling:** Demonstrates the `autogen_ext` Model Context Protocol (MCP) to interact with external APIs (Alpha Vantage) and the local filesystem.
  * **Automated Reporting:** Generates structured JSON reports for both the personal spending analysis and the real-time portfolio data.

-----

## 🔧 Technical Stack

  * **Core:** Python 3.10+
  * **AI & Agents:** AutoGen (`autogen-agentchat`, `autogen_ext`), LangChain (for SQL Agent), OpenAI (`gpt-4o-mini`)
  * **Data & Analysis:** Pandas, NumPy
  * **Database:** SQLite
  * **Visualization:** Plotly, Seaborn
  * **Cloud & APIs:** Boto3 (AWS SDK for S3), Alpha Vantage
  * **Environment:** `dotenv`

-----

## 📁 Project Modules

This project consists of two primary, independent scripts:

### 1\. `financial_analyst.py` (Personal Spending Analyzer)

This script connects to a local SQLite database (`sept_dataset.db`) to perform a detailed analysis of personal spending habits.

**Key Components:**

  * **`SQLAnomalyDetector`:** A class that loads transaction data and identifies anomalies, specifically focusing on **high-frequency transaction days**.
  * **`analyze_financial_data()`:** The main engine that:
    1.  Loads the entire transaction dataset.
    2.  Calculates overall statistics (total spend, date range, etc.).
    3.  Performs analysis by **category**, **month**, **payment mode**, and **day of the week**.
    4.  Generates `plotly` graphs for each analysis (e.g., scatter plot, bar charts, line chart).
    5.  Uploads all generated graphs as PNGs to a specified **AWS S3 bucket** using `boto3`.
    6.  Runs the `SQLAnomalyDetector` to find unusual activity.
  * **`query_db()`:** A LangChain-powered SQL agent that can answer natural language questions about the database (e.g., "show me my last 10 transactions").
  * **Main Execution:** The script runs the `analyze_financial_data` function directly, saving a complete JSON report of the findings (including S3 keys for the graphs) to a local file.

### 2\. `portfolio_analyst.py` (Real-Time Portfolio Analyst)

This script uses AutoGen's modern MCP-based agents to fetch and report on live financial data for a target company (e.g., MSFT).

**Key Components:**

  * **AutoGen MCP:** Uses `StreamableHttpMcpToolAdapter` to connect to the **Alpha Vantage** API and `StdioServerParams` to connect to a **local filesystem** server.
  * **`data_collector` (Agent):** This agent is responsible for calling multiple Alpha Vantage tools to get:
      * Daily time series data
      * Latest news & sentiment
      * Quarterly earnings reports
      * Company overview and key metrics
  * **`report_writer` (Agent):** This agent receives the formatted text and raw data from the `data_collector`. Its sole job is to parse this information into a structured JSON format and use the **filesystem `write_file` tool** to save the final report to the local disk.

-----

## 🛠️ Setup & Installation

1.  **Clone the Repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install Dependencies:**

    ```bash
    # It is recommended to use a virtual environment
    python -m venv venv
    source venv/bin/activate 

    pip install pyautogen autogen-agentchat langchain langchain-openai pandas plotly boto3 python-dotenv numpy seaborn
    ```

3.  **Set Up Environment Variables:**
    Create a `.env` file in the root directory and add your API keys:

    ```ini
    # OpenAI API Key
    openai_api_key="sk-..."

    # Alpha Vantage API Key (for portfolio_analyst.py)
    alphavantage_api_key="YOUR_ALPHA_VANTAGE_KEY"

    # AWS S3 Credentials (for financial_analyst.py)
    S3_ACCESS_KEY="YOUR_AWS_ACCESS_KEY"
    S3_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_KEY"
    S3_BUCKET_NAME="your-s3-bucket-name"
    AWS_REGION_NAME="your-bucket-region" 
    ```

4.  **Database:**
    Ensure you have the `sept_dataset.db` file in the same directory for `financial_analyst.py` to work.

5.  **Node.js (for `portfolio_analyst.py`):**
    The portfolio analyst requires a Node.js-based MCP server for filesystem access. Ensure you have [Node.js](https://nodejs.org/) installed.

-----

## 🏃 How to Run

### 1\. Personal Spending Analysis

Simply run the Python script.

```bash
python financial_analyst.py
```

This will:

  * Connect to `sept_dataset.db`.
  * Perform the full data analysis.
  * Upload `png` graphs to your S3 bucket.
  * Save a `financial_analysis_report_YYYYMMDD_HHMMSS.json` file in the same directory.

### 2\. Real-Time Portfolio Analysis

This is a two-step process involving two terminals.

**Terminal 1: Start the Filesystem MCP Server**
(Run this from your project's root directory)

```bash
npx -y @modelcontextprotocol/server-filesystem .
```

This starts a local server that AutoGen can use to write files.

**Terminal 2: Run the Portfolio Analyst Script**

```bash
python portfolio_analyst.py
```

This will:

  * Trigger the `data_collector` agent to fetch data from Alpha Vantage.
  * Pass the data to the `report_writer` agent.
  * Save a JSON report (e.g., `MSFT_financial_report_...json`) to your local directory.

-----

## 🔮 Future Work

  * **Unify Agents:** Combine both scripts into a single application where a "master" agent can delegate tasks to either the spending analyst or the portfolio analyst.
  * **Re-introduce Advanced ML:** Integrate more robust anomaly detection models (like Isolation Forest, Z-Score) that were planned in the original project.
  * **Prediction:** Implement ML-based forecasting to predict user spending and help create budgets.
  * **Web Interface:** Build a simple Streamlit or Gradio front-end to interact with the agents and view the reports and graphs.
