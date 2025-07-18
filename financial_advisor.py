# -*- coding: utf-8 -*-

pip install -r requirements.txt

import pandas as pd
import sqlite3
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
import asyncio
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

model_client = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key="Your-API-Key-Here",
    temperature=0.1,  # Slightly higher temperature
    max_tokens=1000   # Ensure enough tokens for responses
)

db = SQLDatabase.from_uri("sqlite:///mydb.db")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

sql_agent = create_sql_agent(
        llm=model_client,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-functions",
        handle_parsing_errors=True,
        max_iterations=20,
        early_stopping_method="generate"
    )

def query_db(query: str) -> str:
    """
    Query the database using the SQL agent.

    Args:
        query (str): The user's question or request about the data

    Returns:
        str: The result from the database query
    """
    try:
        # Add context about the transactions table
        enhanced_query = f"{query}. Use the transactions table for this query."
        result = sql_agent.run(enhanced_query)
        return str(result)
    except Exception as e:
        return f"Error querying database: {str(e)}"

class SQLAnomalyDetector:
    def __init__(self, db_path="mydb.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.df = None

    def load_data(self, query=None):
        """Load data from SQL database with improved date/time handling"""
        if query is None:
            query = "SELECT * FROM transactions"

        self.df = pd.read_sql_query(query, self.conn)

        # Print column information for debugging
        print(f"Loaded {len(self.df)} records")
        print(f"Columns: {list(self.df.columns)}")
        if len(self.df) > 0:
            print(f"Column types: {self.df.dtypes.to_dict()}")

        # Handle your specific date/time format
        datetime_created = False

        # Your data has 'Date' and 'Time' columns in format: '01-Mar-22' and '12:00 AM'
        if 'Date' in self.df.columns and 'Time' in self.df.columns:
            try:
                print("Attempting to combine Date and Time columns...")
                if len(self.df) > 0:
                    print(f"Sample Date: {self.df['Date'].iloc[0]}")
                    print(f"Sample Time: {self.df['Time'].iloc[0]}")

                # Combine date and time strings
                combined_strings = self.df['Date'].astype(str) + ' ' + self.df['Time'].astype(str)

                # Try different parsing approaches for your specific format
                try:
                    # First try: assume format like '01-Mar-22 12:00 AM'
                    self.df['DateTime'] = pd.to_datetime(combined_strings, format='%d-%b-%y %I:%M %p', errors='coerce')
                except:
                    try:
                        # Second try: let pandas infer the format
                        self.df['DateTime'] = pd.to_datetime(combined_strings, errors='coerce')
                    except:
                        # Third try: manual parsing
                        self.df['DateTime'] = pd.to_datetime(combined_strings, infer_datetime_format=True, errors='coerce')

                # Check if conversion was successful
                valid_dates = self.df['DateTime'].notna().sum()
                if valid_dates > 0:
                    datetime_created = True
                    print(f"✓ Successfully created DateTime from Date and Time columns")
                    print(f"✓ {valid_dates}/{len(self.df)} valid dates created")
                    print(f"Date range: {self.df['DateTime'].min()} to {self.df['DateTime'].max()}")
                else:
                    print("✗ Failed to parse Date and Time columns")

            except Exception as e:
                print(f"Error combining Date and Time: {e}")

        # Fallback: try to find other datetime columns
        if not datetime_created:
            print("⚠ Could not create DateTime from Date/Time columns")
            print("⚠ Time-based anomaly detection will be skipped")

        return self.df

    def statistical_outliers(self, column, method='z_score', threshold=3):
        """Detect outliers using statistical methods"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in data")

        data = self.df[column].dropna()

        if len(data) == 0:
            return pd.DataFrame()

        # Convert to numeric if possible
        try:
            data = pd.to_numeric(data, errors='coerce').dropna()
        except:
            pass

        if len(data) == 0:
            print(f"Warning: No numeric data found in column '{column}'")
            return pd.DataFrame()

        if method == 'z_score':
            if data.std() == 0:
                return pd.DataFrame()
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = z_scores > threshold

        elif method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                return pd.DataFrame()
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (data < lower_bound) | (data > upper_bound)

        elif method == 'modified_z_score':
            median = data.median()
            mad = np.median(np.abs(data - median))
            if mad == 0:
                return pd.DataFrame()
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = np.abs(modified_z_scores) > threshold

        return self.df[self.df.index.isin(data[outliers].index)]

    def isolation_forest_detection(self, columns, contamination=0.1):
        """Isolation Forest - Good for high-dimensional data"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Check if columns exist
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")

        # Select only numeric columns
        numeric_cols = []
        for col in columns:
            try:
                numeric_data = pd.to_numeric(self.df[col], errors='coerce')
                if numeric_data.notna().sum() > 0:
                    numeric_cols.append(col)
            except:
                continue

        if not numeric_cols:
            print("Warning: No numeric columns found for isolation forest")
            return pd.DataFrame()

        X = self.df[numeric_cols].copy()

        # Convert to numeric and handle missing values
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        X = X.dropna()

        if len(X) == 0:
            return pd.DataFrame()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(X_scaled)

        anomaly_mask = outliers == -1
        return self.df.loc[X.index[anomaly_mask]]

    def financial_transaction_anomalies(self):
        """Detect financial transaction specific anomalies"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        anomalies = {}

        # Find amount columns with different possible names
        
        amount_cols = []
        for col in ['Amount', 'amount', 'AMOUNT', 'Cash Out', 'cash_out', 'Cash In', 'cash_in']:
            if col in self.df.columns:
                amount_cols.append(col)

        # 1. Large cash withdrawals
        for col in ['Cash Out', 'cash_out', 'withdrawal', 'debit']:
            if col in self.df.columns:
                try:
                    cash_out_data = pd.to_numeric(self.df[col], errors='coerce')
                    if cash_out_data.notna().sum() > 0 and (cash_out_data > 0).sum() > 0:
                        large_withdrawals = self.statistical_outliers(col, method='iqr')
                        if len(large_withdrawals) > 0:
                            anomalies['large_withdrawals'] = large_withdrawals
                        break
                except:
                    continue

        # 2. Large cash deposits
        for col in ['Cash In', 'cash_in', 'deposit', 'credit']:
            if col in self.df.columns:
                try:
                    cash_in_data = pd.to_numeric(self.df[col], errors='coerce')
                    if cash_in_data.notna().sum() > 0 and (cash_in_data > 0).sum() > 0:
                        large_deposits = self.statistical_outliers(col, method='iqr')
                        if len(large_deposits) > 0:
                            anomalies['large_deposits'] = large_deposits
                        break
                except:
                    continue

        # 3. Unusual amounts
        for col in ['Amount', 'amount', 'AMOUNT', 'transaction_amount']:
            if col in self.df.columns:
                try:
                    unusual_amounts = self.statistical_outliers(col, method='iqr')
                    if len(unusual_amounts) > 0:
                        anomalies['unusual_amounts'] = unusual_amounts
                    break
                except:
                    continue

        # 4. Frequency anomalies 
        if 'DateTime' in self.df.columns and self.df['DateTime'].notna().sum() > 0:
            try:
                daily_counts = self.df.groupby(self.df['DateTime'].dt.date).size()
                if len(daily_counts) > 1:
                    freq_threshold = daily_counts.mean() + 2 * daily_counts.std()
                    high_freq_days = daily_counts[daily_counts > freq_threshold]
                    if len(high_freq_days) > 0:
                        high_freq_transactions = self.df[self.df['DateTime'].dt.date.isin(high_freq_days.index)]
                        anomalies['high_frequency_days'] = high_freq_transactions
            except Exception as e:
                print(f"Warning: Could not analyze frequency patterns: {e}")

        # Weekday vs Weekend analysis - only if DateTime exists
        if 'DateTime' in self.df.columns and self.df['DateTime'].notna().sum() > 0:
            try:
                # Add day of week (0=Monday, 6=Sunday)
                self.df['day_of_week'] = self.df['DateTime'].dt.dayofweek
                self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6])  # Saturday=5, Sunday=6
                
                # Count transactions by weekday vs weekend
                weekday_count = (~self.df['is_weekend']).sum()
                weekend_count = self.df['is_weekend'].sum()
                
                # Calculate expected weekend ratio (2/7 ≈ 0.286)
                expected_weekend_ratio = 2/7
                actual_weekend_ratio = weekend_count / (weekday_count + weekend_count)
                
                # If weekend transactions are significantly higher than expected, flag as anomaly
                if actual_weekend_ratio > expected_weekend_ratio * 1.5:  # 50% higher than expected
                    weekend_transactions = self.df[self.df['is_weekend']]
                    anomalies['excessive_weekend_activity'] = weekend_transactions
                
                # Also check for unusual weekend transaction amounts
                if weekend_count > 0 and weekday_count > 0:
                    # Find amount columns and compare weekend vs weekday patterns
                    for col in amount_cols:
                        if col in self.df.columns:
                            try:
                                amounts = pd.to_numeric(self.df[col], errors='coerce')
                                if amounts.notna().sum() > 0:
                                    weekend_amounts = amounts[self.df['is_weekend']].dropna()
                                    weekday_amounts = amounts[~self.df['is_weekend']].dropna()
                                    
                                    if len(weekend_amounts) > 0 and len(weekday_amounts) > 0:
                                        # Compare median amounts
                                        weekend_median = weekend_amounts.median()
                                        weekday_median = weekday_amounts.median()
                                        
                                        # If weekend median is significantly higher, flag transactions
                                        if weekend_median > weekday_median * 1.5:
                                            high_weekend_amounts = self.df[
                                                self.df['is_weekend'] & 
                                                (amounts > weekend_amounts.quantile(0.75))
                                            ]
                                            if len(high_weekend_amounts) > 0:
                                                anomalies['high_weekend_amounts'] = high_weekend_amounts
                                    break
                            except:
                                continue
                                
            except Exception as e:
                print(f"Warning: Could not analyze weekday/weekend patterns: {e}")

        return anomalies

    def comprehensive_analysis(self, amount_col='Amount', cash_in_col='Cash In', cash_out_col='Cash Out'):
        """Run comprehensive anomaly detection analysis for financial transactions"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        results = {}

        print("🔍 Running Comprehensive Financial Anomaly Detection Analysis...")
        print("="*60)

        # Check available columns based on your dataset
        available_cols = []
        column_mapping = {
            'Amount': amount_col,
            'Cash In': cash_in_col,
            'Cash Out': cash_out_col
        }

        for display_name, col_name in column_mapping.items():
            if col_name in self.df.columns:
                available_cols.append(col_name)
                print(f"✓ Found column: {col_name}")
            else:
                print(f"✗ Column not found: {col_name}")

        if not available_cols:
            print(f"❌ None of the specified columns found in data")
            print(f"Available columns: {list(self.df.columns)}")
            return results

        # 1. Amount anomalies (both positive and negative values)
        if amount_col in self.df.columns:
            print(f"\n1. Amount Anomalies Analysis ({amount_col})")
            try:
                # Z-score method
                z_anomalies = self.statistical_outliers(amount_col, method='z_score')
                results['amount_z_score'] = z_anomalies
                print(f"   Z-Score outliers: {len(z_anomalies)} anomalies")

                # IQR method
                iqr_anomalies = self.statistical_outliers(amount_col, method='iqr')
                results['amount_iqr'] = iqr_anomalies
                print(f"   IQR outliers: {len(iqr_anomalies)} anomalies")

                # Show some stats
                amount_data = pd.to_numeric(self.df[amount_col], errors='coerce').dropna()
                if len(amount_data) > 0:
                    print(f"   Amount range: {amount_data.min()} to {amount_data.max()}")
                    print(f"   Amount mean: {amount_data.mean():.2f}, std: {amount_data.std():.2f}")

            except Exception as e:
                print(f"   Error analyzing amounts: {e}")

        # 2. Cash In anomalies (deposits)
        if cash_in_col in self.df.columns:
            print(f"\n2. Cash In Anomalies Analysis ({cash_in_col})")
            try:
                # Only analyze non-zero cash in values
                cash_in_data = pd.to_numeric(self.df[cash_in_col], errors='coerce')
                non_zero_mask = cash_in_data > 0

                if non_zero_mask.sum() > 0:
                    cash_in_anomalies = self.statistical_outliers(cash_in_col, method='iqr')
                    results['cash_in_anomalies'] = cash_in_anomalies
                    print(f"   Found {len(cash_in_anomalies)} large deposit anomalies")
                    print(f"   Non-zero deposits: {non_zero_mask.sum()}/{len(self.df)} transactions")
                else:
                    print(f"   No cash deposits found in {cash_in_col}")

            except Exception as e:
                print(f"   Error analyzing cash in: {e}")

        # 3. Cash Out anomalies (withdrawals)
        if cash_out_col in self.df.columns:
            print(f"\n3. Cash Out Anomalies Analysis ({cash_out_col})")
            try:
                # Only analyze non-zero cash out values
                cash_out_data = pd.to_numeric(self.df[cash_out_col], errors='coerce')
                non_zero_mask = cash_out_data > 0

                if non_zero_mask.sum() > 0:
                    cash_out_anomalies = self.statistical_outliers(cash_out_col, method='iqr')
                    results['cash_out_anomalies'] = cash_out_anomalies
                    print(f"   Found {len(cash_out_anomalies)} large withdrawal anomalies")
                    print(f"   Non-zero withdrawals: {non_zero_mask.sum()}/{len(self.df)} transactions")
                else:
                    print(f"   No cash withdrawals found in {cash_out_col}")

            except Exception as e:
                print(f"   Error analyzing cash out: {e}")

        # 4. Multi-column isolation forest
        print(f"\n4. Multi-dimensional Analysis (Isolation Forest)")
        try:
            numeric_cols = []
            for col in available_cols:
                try:
                    test_data = pd.to_numeric(self.df[col], errors='coerce')
                    if test_data.notna().sum() > 0:
                        numeric_cols.append(col)
                except:
                    continue

            if len(numeric_cols) >= 2:
                iso_anomalies = self.isolation_forest_detection(numeric_cols)
                results['isolation_forest'] = iso_anomalies
                print(f"   Using columns: {numeric_cols}")
                print(f"   Found {len(iso_anomalies)} multi-dimensional anomalies")
            else:
                print(f"   Need at least 2 numeric columns. Found: {numeric_cols}")

        except Exception as e:
            print(f"   Error in isolation forest: {e}")

        # 5. Transaction pattern anomalies
        print(f"\n5. Transaction Pattern Analysis")
        try:
            # Category-based anomalies
            if 'Category' in self.df.columns:
                category_counts = self.df['Category'].value_counts()
                print(f"   Transaction categories: {len(category_counts)}")
                print(f"   Most common: {category_counts.head(3).to_dict()}")

                # Find rare categories
                rare_categories = category_counts[category_counts <= 2]
                if len(rare_categories) > 0:
                    rare_transactions = self.df[self.df['Category'].isin(rare_categories.index)]
                    results['rare_category_transactions'] = rare_transactions
                    print(f"   Rare category transactions: {len(rare_transactions)}")
        except Exception as e:
            print(f"   Error analyzing patterns: {e}")

        # 6. Financial-specific anomalies
        print(f"\n6. Financial-Specific Anomaly Detection")
        try:
            fin_anomalies = self.financial_transaction_anomalies()
            results['financial_anomalies'] = fin_anomalies

            for anomaly_type, anomaly_data in fin_anomalies.items():
                if isinstance(anomaly_data, pd.DataFrame):
                    print(f"   {anomaly_type.replace('_', ' ').title()}: {len(anomaly_data)} anomalies")

        except Exception as e:
            print(f"   Error in financial anomaly detection: {e}")

        # 7. Time-based analysis (if DateTime available)
        if 'DateTime' in self.df.columns and self.df['DateTime'].notna().sum() > 0:
            print(f"\n7. Time-Based Analysis")
            try:
                # Hour-based patterns
                hour_counts = self.df['DateTime'].dt.hour.value_counts().sort_index()
                print(f"   Time range: {hour_counts.index.min()}:00 to {hour_counts.index.max()}:00")

                # Find unusual timing
                unusual_hours = hour_counts[hour_counts <= 1]  # Very few transactions
                if len(unusual_hours) > 0:
                    unusual_time_transactions = self.df[self.df['DateTime'].dt.hour.isin(unusual_hours.index)]
                    results['unusual_time_transactions'] = unusual_time_transactions
                    print(f"   Unusual timing transactions: {len(unusual_time_transactions)}")
                
                # Weekday vs Weekend analysis
                if 'is_weekend' in self.df.columns:
                    weekday_count = (~self.df['is_weekend']).sum()
                    weekend_count = self.df['is_weekend'].sum()
                    weekend_ratio = weekend_count / (weekday_count + weekend_count)
                    print(f"   Weekday transactions: {weekday_count} ({(1-weekend_ratio)*100:.1f}%)")
                    print(f"   Weekend transactions: {weekend_count} ({weekend_ratio*100:.1f}%)")
                    print(f"Weekend ratio(w.r.t. week):{weekend_ratio}")
                    print(f"   Expected weekend ratio: 0.35-0.45")

            except Exception as e:
                print(f"   Error in time analysis: {e}")

        return results

def analyze_financial_data(query: str = None) -> str:
    """
    Analyze financial data using the SQLAnomalyDetector with detailed category, month, and mode splits

    Args:
        query (str): Optional SQL query to filter data

    Returns:
        str: Formatted analysis results with detailed breakdowns
    """
    try:
        detector = SQLAnomalyDetector("mydb.db")

        # Load data
        df = detector.load_data(query)

        if df is None or len(df) == 0:
            return "No data found for analysis"

        # Filter for positive amounts only (expenses/spending)
        df_positive = df[df['Amount'] > 0].copy()
        
        if len(df_positive) == 0:
            return "No positive amount transactions found for analysis"

        # Run comprehensive analysis on positive amounts only
        results = detector.comprehensive_analysis()

        # Initialize detailed analysis string to capture all insights
        detailed_analysis = f"""
📊 COMPREHENSIVE FINANCIAL DATA ANALYSIS (Positive Amounts Only)
================================================================

Dataset Overview:
- Total records (all): {len(df)}
- Positive amount records: {len(df_positive)}
- Date range: {df_positive['Date'].min() if 'Date' in df_positive.columns else 'N/A'} to {df_positive['Date'].max() if 'Date' in df_positive.columns else 'N/A'}
- Columns: {list(df_positive.columns)}
"""

        # Use df_positive for all subsequent calculations
        df = df_positive.copy()

        # === CATEGORY ANALYSIS ===
        if 'Category' in df.columns:
            detailed_analysis += "\n\n🏷️ CATEGORY BREAKDOWN:\n" + "="*50 + "\n"
            
            # category-counts split
            category_stats = df.groupby('Category').agg({
                'Amount': ['count', 'sum', 'mean', 'std', 'min', 'max']
            }).round(2)
            category_count_df = category_stats['Amount']['count'].reset_index()
            category_count_df.columns = ['Category', 'Transaction Count']

            # category-amount splits (now all amounts are positive)
            if 'Amount' in df.columns:
                # Remove the Money category filter since we're only dealing with positive amounts
                category_amounts = df.groupby('Category')['Amount'].agg(['sum', 'mean', 'count']).round(2)
                category_amounts = category_amounts.sort_values('sum', ascending=False)
                
                # Capture the insights
                detailed_analysis += f"\n   Amount spent by category:\n"
                detailed_analysis += f"   Top 5 categories by total amount:\n"
                for cat, row in category_amounts.head(5).iterrows():
                    detailed_analysis += f"     {cat}: Rs.{row['sum']:,.2f} (avg: Rs.{row['mean']:,.2f}, {row['count']} transactions)\n"
                
                # Create and display plot
                plt.figure(figsize=(12, 8))
                
                # Since all amounts are positive, we can simplify the plotting
                bars = plt.bar(range(len(category_amounts)), 
                              category_amounts['sum'], 
                              color='green', alpha=0.7)
                
                plt.xticks(range(len(category_amounts)), 
                          category_amounts.index, 
                          rotation=45, ha='right')
                plt.title('Total Amount Spent per Category (Positive Amounts Only):', fontsize=16, pad=20)
                plt.ylabel('Total Amount (Rs.)', fontsize=12)
                plt.xlabel('Category', fontsize=12)
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, category_amounts['sum'])):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + (height * 0.01),
                            f'Rs.{value:,.0f}', ha='center', va='bottom',
                            fontsize=10, rotation=0)
                
                plt.tight_layout()
                plt.show()
            
            # Calculate totals for percentages
            total_transactions = len(df)
            total_amount = df['Amount'].sum() if 'Amount' in df.columns else 0
            
            detailed_analysis += f"\nTotal Categories: {len(category_stats)}\n"
            detailed_analysis += f"Total Transactions: {total_transactions}\n"
            detailed_analysis += f"Total Amount: Rs.{total_amount:,.2f}\n\n"
            
            # Update column names for the stats
            category_stats.columns = ['Count', 'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Min_Amount', 'Max_Amount']
            
            detailed_analysis += "📈 Category Statistics:\n"
            for category, stats in category_stats.iterrows():
                percentage = (stats['Count'] / total_transactions) * 100
                amount_percentage = (stats['Total_Amount'] / total_amount) * 100 if total_amount > 0 else 0
                detailed_analysis += f"\n  {category}:\n"
                detailed_analysis += f"    • Transactions: {stats['Count']} ({percentage:.1f}% of total)\n"
                detailed_analysis += f"    • Total Amount: Rs.{stats['Total_Amount']:,.2f} ({amount_percentage:.1f}% of total)\n"
                detailed_analysis += f"    • Average Amount: Rs.{stats['Avg_Amount']:,.2f}\n"
                detailed_analysis += f"    • Amount Range: Rs.{stats['Min_Amount']:,.2f} to Rs.{stats['Max_Amount']:,.2f}\n"
                detailed_analysis += f"    • Std Deviation: Rs.{stats['Std_Amount']:,.2f}\n"

        # === MONTHLY ANALYSIS ===
        if 'Date' in df.columns:
            detailed_analysis += "\n\n📅 MONTHLY BREAKDOWN:\n" + "="*50 + "\n"
            
            try:
                # Convert Date to datetime and extract month-year
                df['Date_parsed'] = pd.to_datetime(df['Date'], format='%d-%b-%y', errors='coerce')
                df['Month_Year'] = df['Date_parsed'].dt.to_period('M')
                
                monthly_stats = df.groupby('Month_Year').agg({
                    'Amount': ['count', 'sum', 'mean', 'std', 'min', 'max']
                }).round(2)
                
                monthly_stats.columns = ['Count', 'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Min_Amount', 'Max_Amount']
                monthly_stats = monthly_stats.sort_index()
                
                detailed_analysis += f"Date Range: {monthly_stats.index.min()} to {monthly_stats.index.max()}\n"
                detailed_analysis += f"Total Months: {len(monthly_stats)}\n\n"
                
                detailed_analysis += "📊 Monthly Statistics:\n"
                for month, stats in monthly_stats.iterrows():
                    detailed_analysis += f"\n  {month}:\n"
                    detailed_analysis += f"    • Transactions: {stats['Count']}\n"
                    detailed_analysis += f"    • Total Amount: Rs.{stats['Total_Amount']:,.2f}\n"
                    detailed_analysis += f"    • Average Amount: Rs.{stats['Avg_Amount']:,.2f}\n"
                    detailed_analysis += f"    • Amount Range: Rs.{stats['Min_Amount']:,.2f} to Rs.{stats['Max_Amount']:,.2f}\n"
                    detailed_analysis += f"    • Std Deviation: Rs.{stats['Std_Amount']:,.2f}\n"
                
                # Monthly trends
                detailed_analysis += "\n📈 Monthly Trends:\n"
                avg_monthly_amount = monthly_stats['Total_Amount'].mean()
                highest_month = monthly_stats['Total_Amount'].idxmax()
                lowest_month = monthly_stats['Total_Amount'].idxmin()
                
                detailed_analysis += f"  • Average Monthly Total: Rs.{avg_monthly_amount:,.2f}\n"
                detailed_analysis += f"  • Highest Activity: {highest_month} (Rs.{monthly_stats.loc[highest_month, 'Total_Amount']:,.2f})\n"
                detailed_analysis += f"  • Lowest Activity: {lowest_month} (Rs.{monthly_stats.loc[lowest_month, 'Total_Amount']:,.2f})\n"
                
            except Exception as e:
                detailed_analysis += f"Error processing monthly data: {str(e)}\n"

        # === PAYMENT MODE ANALYSIS ===
        if 'Mode' in df.columns:
            detailed_analysis += "\n\n💳 PAYMENT MODE BREAKDOWN:\n" + "="*50 + "\n"
            
            mode_stats = df.groupby('Mode').agg({
                'Amount': ['count', 'sum', 'mean', 'std', 'min', 'max']
            }).round(2)
            
            mode_stats.columns = ['Count', 'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Min_Amount', 'Max_Amount']
            mode_stats = mode_stats.sort_values('Total_Amount', ascending=False)
            
            detailed_analysis += f"Total Payment Modes: {len(mode_stats)}\n"
            detailed_analysis += f"Most Used Mode: {mode_stats.index[0]} ({mode_stats.iloc[0]['Count']} transactions)\n"
            detailed_analysis += f"Highest Value Mode: {mode_stats['Total_Amount'].idxmax()}\n\n"
            
            detailed_analysis += "💰 Payment Mode Statistics:\n"
            for mode, stats in mode_stats.iterrows():
                percentage = (stats['Count'] / total_transactions) * 100
                amount_percentage = (stats['Total_Amount'] / total_amount) * 100 if total_amount > 0 else 0
                detailed_analysis += f"\n  {mode}:\n"
                detailed_analysis += f"    • Transactions: {stats['Count']} ({percentage:.1f}% of total)\n"
                detailed_analysis += f"    • Total Amount: Rs.{stats['Total_Amount']:,.2f} ({amount_percentage:.1f}% of total)\n"
                detailed_analysis += f"    • Average Amount: Rs.{stats['Avg_Amount']:,.2f}\n"
                detailed_analysis += f"    • Amount Range: Rs.{stats['Min_Amount']:,.2f} to Rs.{stats['Max_Amount']:,.2f}\n"
                detailed_analysis += f"    • Std Deviation: Rs.{stats['Std_Amount']:,.2f}\n"

        # === CROSS-ANALYSIS ===
        detailed_analysis += "\n\n🔄 CROSS-ANALYSIS:\n" + "="*50 + "\n"
        
        # Category vs Mode analysis
        if 'Category' in df.columns and 'Mode' in df.columns:
            detailed_analysis += "🏷️💳 Category vs Payment Mode:\n"
            cross_analysis = df.groupby(['Category', 'Mode']).agg({
                'Amount': ['count', 'sum', 'mean']
            }).round(2)
            
            cross_analysis.columns = ['Count', 'Total_Amount', 'Avg_Amount']
            
            for category in df['Category'].unique():
                if category in cross_analysis.index:
                    cat_data = cross_analysis.loc[category]
                    if isinstance(cat_data, pd.Series):
                        cat_data = cat_data.to_frame().T
                    if len(cat_data) > 0:
                        detailed_analysis += f"\n  {category}:\n"
                        for mode_idx, stats in cat_data.iterrows():
                            mode_name = mode_idx if isinstance(mode_idx, str) else mode_idx
                            detailed_analysis += f"    • {mode_name}: {stats['Count']} transactions, Total: Rs.{stats['Total_Amount']:,.2f}\n"

        # === ANOMALY DETECTION RESULTS ===
        detailed_analysis += "\n\n🚨 ANOMALY DETECTION RESULTS:\n" + "="*50 + "\n"
        
        total_anomalies = 0
        for key, value in results.items():
            if isinstance(value, dict):
                if key == 'financial_anomalies':
                    detailed_analysis += f"\n{key.replace('_', ' ').title()}:\n"
                    for anomaly_type, anomaly_data in value.items():
                        if isinstance(anomaly_data, pd.DataFrame):
                            anomaly_count = len(anomaly_data)
                            total_anomalies += anomaly_count
                            detailed_analysis += f"  • {anomaly_type.replace('_', ' ').title()}: {anomaly_count} anomalies\n"
                            
                            # Show sample anomalies
                            if anomaly_count > 0:
                                detailed_analysis += "    Sample anomalies:\n"
                                for i, (idx, row) in enumerate(anomaly_data.head(3).iterrows()):
                                    detailed_analysis += f"      Row {idx}: "
                                    if 'Amount' in row:
                                        detailed_analysis += f"Amount: Rs.{row['Amount']:,.2f}, "
                                    if 'Category' in row:
                                        detailed_analysis += f"Category: {row['Category']}, "
                                    if 'Mode' in row:
                                        detailed_analysis += f"Mode: {row['Mode']}"
                                    detailed_analysis += "\n"
                        
                elif 'mean' in str(value):
                    detailed_analysis += f"\n{key.replace('_', ' ').title()}:\n"
                    if hasattr(value, 'items'):
                        for stat_key, stat_value in value.items():
                            detailed_analysis += f"  • {stat_key}: {stat_value}\n"
                            
            elif isinstance(value, pd.DataFrame):
                anomaly_count = len(value)
                total_anomalies += anomaly_count
                detailed_analysis += f"\n{key.replace('_', ' ').title()}: {anomaly_count} anomalies detected\n"
                
                if anomaly_count > 0:
                    detailed_analysis += "  Sample anomalies:\n"
                    for i, (idx, row) in enumerate(value.head(3).iterrows()):
                        detailed_analysis += f"    • Row {idx}: "
                        if 'Amount' in row:
                            detailed_analysis += f"Amount: Rs.{row['Amount']:,.2f}, "
                        if 'Category' in row:
                            detailed_analysis += f"Category: {row['Category']}, "
                        if 'Mode' in row:
                            detailed_analysis += f"Mode: {row['Mode']}"
                        detailed_analysis += "\n"

        # === SUMMARY INSIGHTS ===
        detailed_analysis += "\n\n💡 KEY INSIGHTS:\n" + "="*50 + "\n"
        
        if 'Category' in df.columns:
            top_category = df['Category'].value_counts().index[0]
            detailed_analysis += f"• Most frequent category: {top_category}\n"
            
        if 'Mode' in df.columns:
            top_mode = df['Mode'].value_counts().index[0]
            detailed_analysis += f"• Most used payment mode: {top_mode}\n"
            
        if 'Amount' in df.columns:
            avg_amount = df['Amount'].mean()
            detailed_analysis += f"• Average transaction amount: Rs.{avg_amount:,.2f}\n"
            
        detailed_analysis += f"• Total anomalies detected: {total_anomalies}\n"

        # Create spending over time plot (now only positive amounts)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y', errors='coerce')
        daily = df.set_index('Date').resample('D')['Amount'].sum().fillna(0)
        
        plt.figure(figsize=(12, 5))
        sns.lineplot(data=daily)
        plt.title('Daily Spending Over Time (Positive Amounts Only)')
        plt.ylabel('Daily Spending Amount (Rs.)')
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return detailed_analysis

    except Exception as e:
        return f"Error in financial analysis: {str(e)}"



def finance_analysis(user_query: str) -> str:
    """
    Combined finance analysis function that queries database and performs analysis

    Args:
        user_query (str): User's financial query

    Returns:
        str: Comprehensive financial analysis and recommendations
    """
    try:
        print("📊 Starting Financial Analysis...")

        # Step 1: Query database for transaction data
        print("🔍 Step 1: Querying transaction database...")
        db_query = "SELECT * FROM transactions"
        db_result = query_db(db_query)

        # Step 2: Perform comprehensive statistical analysis
        print("📈 Step 2: Performing comprehensive statistical analysis...")
        analysis_result = analyze_financial_data()

        # Step 3: Combine results and provide financial insights
        print("💡 Step 3: Generating financial insights...")

        combined_analysis = f"""
🏦 COMPREHENSIVE FINANCIAL ANALYSIS REPORT
===========================================

USER QUERY: {user_query}

{analysis_result}

💰 FINANCIAL INSIGHTS & RECOMMENDATIONS:
==========================================

Based on the statistical analysis of your transaction data, here are the key findings and recommendations:

1. **Spending Patterns**: Your transaction data shows various spending categories including Food, Travel, Personal expenses, and regular income sources.

2. **Payment Preferences**: The analysis reveals your preferred payment methods and their usage patterns.

3. **Anomaly Detection**: Statistical outliers have been identified that may represent unusual spending or income patterns worth reviewing.

4. **Budget Recommendations**:
   - Review high-value outlier transactions to ensure they align with your financial goals
   - Consider the distribution of spending across categories for better budget allocation
   - Monitor payment mode preferences for potential optimization

5. **Risk Assessment**: The presence of anomalies suggests areas where you might want to implement spending controls or alerts.

🎯 NEXT STEPS:
- Review the statistical outliers identified in the analysis
- Consider setting up budget categories based on your spending patterns
- Monitor transactions that deviate significantly from your normal patterns
- Implement alerts for unusual spending behaviors

This analysis provides a foundation for better financial decision-making and budget management.
"""

        return combined_analysis

    except Exception as e:
        return f"Error in finance analysis: {str(e)}"

finance_agent = AssistantAgent(
    name="finance_agent",
    model_client=model_client,
    tools=[query_db],
    system_message="""You are a senior financial advisor and analyst. Your role is to:

1. **Assess Query Complexity**: Determine if the user query requires basic data retrieval or complex statistical analysis
2. **Handle Basic Queries**: For simple data requests, use query_db to get information and provide basic financial insights
3. **Coordinate Complex Analysis**: For complex queries requiring statistical analysis, anomaly detection, or pattern recognition:
   - First use query_db to understand the data structure
   - Then request the data analyst to perform comprehensive analysis
   - Wait for data analyst's statistical insights
   - Combine their findings with additional data queries if needed
4. **Provide Final Recommendations**: Give actionable financial advice based on SPECIFIC STATISTICAL FINDINGS

**WHEN TO INVOLVE DATA ANALYST**:
- User asks for "analysis", "patterns", "trends", "anomalies", or "insights"
- Requests about spending behavior, risk assessment, or budget optimization
- Questions requiring statistical analysis or data science techniques
- Complex queries that need more than basic data retrieval

**BASIC QUERIES** (handle yourself with query_db):
- "Show me recent transactions"
- "What's my current balance?"
- "List transactions by category"
- Simple data retrieval requests

**COMPLEX QUERIES** (involve data analyst):
- "Analyze my spending patterns"
- "Find unusual transactions"
- "What are my financial trends?"
- "Provide budget recommendations"

**CRITICAL - USE SPECIFIC DATA FROM ANALYSIS**:
When data analyst provides statistical insights, use the ACTUAL NUMBERS and SPECIFIC FINDINGS:
- Reference exact amounts, percentages, and counts from their analysis
- Quote specific statistics like "Your average spending is Rs.X with Rs.Y standard deviation"
- Use actual category breakdowns like "Food represents 270 out of 564 transactions (48%)"
- Reference specific anomaly counts and what they mean financially
- Base recommendations on the actual data patterns found

**IMPORTANT**:
- After data analyst provides statistical insights, interpret the SPECIFIC NUMBERS and provide targeted recommendations
- Don't provide generic advice - use the actual statistical findings
- Always provide financial interpretation and recommendations based on real data
- Reference specific anomalies, spending patterns, and statistical measures

**YOUR FINAL RESPONSE SHOULD INCLUDE**:
- Interpretation of specific statistical findings (using actual numbers)
- Targeted recommendations based on the real spending patterns found
- Risk assessment based on actual anomaly counts and patterns
- Specific next steps based on the data analysis results""",
    reflect_on_tool_use=True,
)

data_analyst = AssistantAgent(
    name="data_analyst",
    model_client=model_client,
    tools=[analyze_financial_data],
    system_message="""You are a senior data analyst specializing in financial data analysis. Your role is to:

1. **Receive Analysis Requests**: Finance agent will request statistical analysis for complex queries
2. **Perform Comprehensive Analysis**: Use analyze_financial_data function to run detailed statistical analysis
3. **Provide Statistical Insights**: Return detailed findings with ACTUAL NUMBERS and SPECIFIC DATA from the analysis

**YOUR ANALYSIS MUST INCLUDE ACTUAL VALUES FROM THE ANALYSIS RESULT**:
- Use the specific numbers returned by the analyze_financial_data function
- Quote exact statistics (mean, median, std dev, ranges) from the analysis
- Reference specific category counts and payment mode statistics
- Include actual anomaly counts and outlier details
- Provide real insights based on the returned data, not generic placeholders

**EXAMPLE OF PROPER ANALYSIS**:
"Based on the analysis results:
- Amount statistics: Mean: -0.49, Std Dev: 3826.63, Range: -61100 to 65044
- Transaction categories: 14 total, with Food (270 transactions), Drink (119), Travel (39) as top categories
- Payment modes: Google Pay (436 transactions), Cash (56), Paytm (30)
- Anomalies detected: 63 IQR outliers, 55 multi-dimensional anomalies, 38 cash withdrawal anomalies"

**IMPORTANT**:
- NEVER use placeholders like "(Specify Mean)" or "(Provide descriptive stats here)"
- ALWAYS extract and report the actual numerical values from the analysis results
- Base your insights on the real data returned by the analyze_financial_data function
- Be specific about what the numbers mean statistically

**RESPONSE FORMAT**:
After analysis, provide comprehensive statistical findings with ACTUAL DATA and conclude with:
"Finance agent, please interpret these SPECIFIC statistical insights and provide financial recommendations to the user."
""",
    reflect_on_tool_use=True,
)

from autogen_agentchat.teams import SelectorGroupChat

group_chat = SelectorGroupChat(
    participants=[finance_agent, data_analyst],
    model_client=model_client,
)

async def run_finance_analysis(user_query: str):
    """
    Run the two-agent financial analysis workflow with live conversation display
    """
    try:
        print("🚀 Starting Two-Agent Financial Analysis...")
        print("="*60)

        task = f"""
USER QUERY: {user_query}

WORKFLOW INSTRUCTIONS:
1. Finance Agent: Assess if this is a basic query or requires complex analysis
2. For BASIC queries: Finance agent uses query_db and provides simple insights
3. For COMPLEX queries:
   - Finance agent requests data analyst to perform statistical analysis
   - Data analyst uses analyze_financial_data function and returns statistical insights
   - Finance agent MUST interpret the statistical insights and provide final financial recommendations

GOAL: Provide comprehensive financial analysis with actionable recommendations.
"""

        # Create an async iterator to show conversation in real-time
        async for message in group_chat.run_stream(task=task):
            if hasattr(message, 'source') and hasattr(message, 'content'):
                if message.source == 'finance_agent':
                    print(f"\n🏦 FINANCE AGENT:")
                    print("-" * 40)
                    print(f"{message.content}")
                    print("-" * 40)
                elif message.source == 'data_analyst':
                    print(f"\n📊 DATA ANALYST:")
                    print("-" * 40)
                    print(f"{message.content}")
                    print("-" * 40)

        print("\n" + "="*60)
        print("✅ Analysis Complete!")
        return True

    except Exception as e:
        print(f"❌ Error in financial analysis: {e}")
        return False

async def main():
    test_queries = [
        "Analyze my spending patterns and provide budget recommendations",  # Complex
        "Show me my recent transactions",  # Basic
        "What are my financial trends and any anomalies?",  # Complex
        "List my transactions by category",  # Basic
        "Perform comprehensive analysis of my financial data",  # Complex
        "What's my current balance trend?"  # Basic
    ]

    query = input("What would you like to know about your financial data? ") or test_queries[0]

    success = await run_finance_analysis(query)

    if not success:
        print("❌ Analysis failed. Please check your setup and try again.")

await main()
