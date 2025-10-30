Financial Insights & Portfolio Analysis System
This project provides a comprehensive suite for financial analysis, split into two main components:

A Personal Spending Analyzer that performs Exploratory Data Analysis (EDA) on a local SQL database of transactions, generates visualizations, and uploads them to AWS S3.

A Real-Time Portfolio Analyst that uses AutoGen's Model Context Protocol (MCP) to fetch live market data from Alpha Vantage for a specific stock and generate a detailed report.

🚀 Core Features
Personal Spending Analysis: Conducts deep EDA on transaction data from a SQLite database.

Real-Time Portfolio Analysis: Fetches live market data, news, earnings, and fundamentals using the Alpha Vantage API.

Multi-Agent Architecture: Uses AutoGen agents for task orchestration, including a two-agent MCP workflow (data_collector -> report_writer).

Natural Language Database Query: Integrates a LangChain SQL Agent (query_db) to answer natural language questions about personal spending data.

Cloud Integration: Automatically uploads generated Plotly graphs (e.g., spending by category, daily trends) to an AWS S3 bucket.

Modern AutoGen Tooling: Demonstrates the autogen_ext Model Context Protocol (MCP) to interact with external APIs (Alpha Vantage) and the local filesystem.

Automated Reporting: Generates structured JSON reports for both the personal spending analysis and the real-time portfolio data.

🔧 Technical Stack
Core: Python 3.10+

AI & Agents: AutoGen (autogen-agentchat, autogen_ext), LangChain (for SQL Agent), OpenAI (gpt-4o-mini)

Data & Analysis: Pandas, NumPy

Database: SQLite

Visualization: Plotly, Seaborn

Cloud & APIs: Boto3 (AWS SDK for S3), Alpha Vantage

Environment: dotenv

📁 Project Modules
This project consists of two primary, independent scripts:

1. financial_analyst.py (Personal Spending Analyzer)
This script connects to a local SQLite database (sept_dataset.db) to perform a detailed analysis of personal spending habits.

Key Components:

SQLAnomalyDetector: A class that loads transaction data and identifies anomalies, specifically focusing on high-frequency transaction days.

analyze_financial_data(): The main engine that:

Loads the entire transaction dataset.

Calculates overall statistics (total spend, date range, etc.).

Performs analysis by category, month, payment mode, and day of the week.

Generates plotly graphs for each analysis (e.g., scatter plot, bar charts, line chart).

Uploads all generated graphs as PNGs to a specified AWS S3 bucket using boto3.

Runs the SQLAnomalyDetector to find unusual activity.

query_db(): A LangChain-powered SQL agent that can answer natural language questions about the database (e.S., "show me my last 10 transactions").

Main Execution: The script runs the analyze_financial_data function directly, saving a complete JSON report of the findings (including S3 keys for the graphs) to a local file.

2. portfolio_analyst.py (Real-Time Portfolio Analyst)
This script uses AutoGen's modern MCP-based agents to fetch and report on live financial data for a target company (e.g., MSFT).

Key Components:

AutoGen MCP: Uses StreamableHttpMcpToolAdapter to connect to the Alpha Vantage API and StdioServerParams to connect to a local filesystem server.

data_collector (Agent): This agent is responsible for calling multiple Alpha Vantage tools to get:

Daily time series data

Latest news & sentiment

Quarterly earnings reports

Company overview and key metrics

report_writer (Agent): This agent receives the formatted text and raw data from the data_collector. Its sole job is to parse this information into a structured JSON format and use the filesystem write_file tool to save the final report to the local disk.
