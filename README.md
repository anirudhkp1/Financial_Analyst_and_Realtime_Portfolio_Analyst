# Autogen Financial_Advisor
Multi-agent Autogen application to analyze historic user data to provide insights on spending patterns.
This project aims to provide insights into user spending patterns, answer user queries, and to help users better track their financial transactions. 

## 🚀 Features

- 🔗 SQL agent integration with LangChain
- 🤖 OpenAI GPT-based assistant via AutoGen
- 📊 Anomaly detection using:
  - Isolation Forest
  - DBSCAN
  - One-Class SVM
  - etc
- 🖼️ Mutli-Agent Architecture using SelectorGroupChat

## Agent Structure

🏦 **finance_agent – Senior Financial Advisor**
Role: Acts as the user-facing assistant responsible for interpreting financial queries and delivering actionable recommendations.

Responsibilities:

Query Evaluation: Determines whether a query is simple (e.g., "Show my last 10 transactions") or complex (e.g., "Analyze my spending trends").

Simple Queries: Directly uses the query_db tool to retrieve and respond to basic data requests.

Complex Queries: Coordinates with data_analyst by:

Using query_db to understand the data

Delegating statistical analysis

Interpreting real statistical results for actionable financial advice

Outputs: Financial insights grounded in actual numbers (e.g., "Your food expenses make up 48% of your total spend").

📊 **data_analyst – Statistical Analyst for Financial Data**
Role: Performs rigorous data analysis to support complex queries with concrete statistical findings.

Responsibilities:

Uses analyze_financial_data to compute detailed statistics

Provides exact values like:

Means, standard deviations, and ranges of transactions

Category and payment method breakdowns

Anomaly counts and types (e.g., IQR, multidimensional)

Avoids generic or placeholder statements — only reports real, computed values

Returns findings to the finance_agent for interpretation and user-facing recommendation

## SQL Agent 

The SQL Agent is a LangChain-powered agent that allows natural language interaction with a SQL database.

Purpose:
To translate user queries into executable SQL statements and return results in a structured, human-readable format.

Key Capabilities:

Understands user intent via natural language

Uses LangChain’s SQLDatabaseToolkit and create_sql_agent for query execution

Works seamlessly with AutoGen agents (e.g., finance_agent) to retrieve data from a SQLite database

Supports flexible, conversational access to structured financial data (e.g., transactions, balances, categories)

Example Queries It Can Handle:

"List all transactions above ₹5000"

"Show spending by category in June"

"Get the average amount spent on travel"

## Future Work

- ML-based prediction algorithm to predict user spending and create budgets
- Improve efficiency and output of final summary from finance agent


