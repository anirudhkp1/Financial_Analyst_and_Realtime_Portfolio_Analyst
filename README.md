# Autogen-based Multi-Agent Financial Advisor

This project aims to provide insights into user spending patterns, answer user queries, and to help users better track their financial transactions. The program consists of 2 agents- a financial advisor and a data analyst. Simple user queries are answered directly by financial agent, whereas more complex queries which require data analysis are passed on to the data analyst agent for categorization and ML-based anomaly detection.

## 🚀 Features

- 🔗 SQL agent integration with LangChain
- 🤖 OpenAI GPT-based assistant via AutoGen
- 📊 Anomaly detection using:
  - Isolation Forest
- 🖼️ Mutli-Agent Architecture using SelectorGroupChat

## Agent Structure

🏦 **finance_agent – Senior Financial Advisor**

Role: Acts as the user-facing assistant responsible for interpreting financial queries and delivering actionable recommendations.

Responsibilities:

-Query Evaluation: Determines whether a query is simple (e.g., "Show my last 10 transactions") or complex (e.g., "Analyze my spending trends").

-Simple Queries: Directly uses the query_db tool to retrieve and respond to basic data requests.

-Complex Queries: Coordinates with data_analyst by:

-Using query_db to understand the data

-Delegating statistical analysis

-Interpreting real statistical results for actionable financial advice

Outputs: Financial insights grounded in actual numbers (e.g., "Your food expenses make up 48% of your total spend").

📊 **data_analyst – Statistical Analyst for Financial Data**

Role: Performs rigorous data analysis to support complex queries with concrete statistical findings.

Responsibilities:

-Uses analyze_financial_data to compute detailed statistics

-Provides exact values like:

  -Means, standard deviations, and ranges of transactions

-Category and payment method breakdowns

-Anomaly counts and types (e.g., IQR, multidimensional)

-Returns findings to the finance_agent for interpretation and user-facing recommendation

## SQL Agent 

The SQL Agent is a LangChain-powered agent that allows natural language interaction with a SQL database.

Purpose:
To translate user queries into executable SQL statements and return results in a structured, human-readable format.

Key Capabilities:

-Understands user intent via natural language

-Uses LangChain’s SQLDatabaseToolkit and create_sql_agent for query execution

-Works seamlessly with AutoGen agents (e.g., finance_agent) to retrieve data from a SQLite database

-Supports flexible, conversational access to structured financial data (e.g., transactions, balances, categories)

-Example Queries It Can Handle:

  -"List all transactions above ₹5000"

  -"Show spending by category in June"
  
  -"Get the average amount spent on travel"

## Data Analysis- Categorization and Anomaly Detection using SQLAnomalyDetector class

📉 SQLAnomalyDetector – Financial Anomaly Detection Engine
The SQLAnomalyDetector class provides a full pipeline for detecting anomalies in financial transaction data loaded from a SQL database.

**🔧 Core Responsibilities**
-Load & Preprocess Data:

  -Connects to a SQLite database and executes SQL queries.
  
  -Combines separate Date and Time columns into a single DateTime field with robust parsing.
  
  -Provides type introspection and handles missing or malformed data gracefully.

-Outlier Detection Methods:

  -Z-Score: Identifies extreme values based on standard deviations.
  
  -IQR (Interquartile Range): Flags values significantly outside the typical range.
  
  -Modified Z-Score: Robust detection using median and MAD (Median Absolute Deviation).
  
  -Isolation Forest: Detects multi-dimensional outliers using scikit-learn’s model.

-🧠 Domain-Specific Anomaly Checks
  -Large Cash Withdrawals/Deposits: Detects significant inflow or outflow of funds using IQR.
  
  -Unusual Transaction Amounts: Flags spikes or dips in transaction values.
  
  -High-Frequency Days: Identifies days with abnormally high numbers of transactions.
  
  -Round Number Bias: Highlights transactions with rounded amounts (e.g., ₹1000, ₹5000) that may be suspicious.
  
  -Late-Night Activity: Detects transactions occurring between 11 PM and 5 AM.
  
  -Rare Categories & Modes: Finds seldom-used transaction categories or payment methods.
  
  -Time-Based Patterns: Analyzes hour-of-day transaction distributions

## Future Work

- ML-based prediction algorithm to predict user spending and create budgets
- Improve efficiency and output of final summary from finance agent


