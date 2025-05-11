"""This script takes FDPS HTML file and parses it to extract relevant data.
@contracts_csv_path: Path to the CSV file containing contract data.
@output_csv_path: Path to the output CSV file where parsed data will be saved.
"""
import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from dateutil import tz
from pathlib import Path
import requests
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
CONTRACTS_CSV_PATH = os.getenv('CONTRACTS_CSV_PATH')
OUTPUT_CSV_PATH = os.getenv('OUTPUT_CSV_PATH') 

# Read in data
df = pd.read_csv(OUTPUT_CSV_PATH, encoding='utf-8')
# Select last 30 days
df['deleted_date'] = pd.to_datetime(df['deleted_date'])
df = df[df['deleted_date'] > (datetime.now() - relativedelta(days=30))]

# ---------------------------------
# Stage 2a: Consistency Checks
# ---------------------------------
# Convert fpds datestrings to datetime objects
df.loc[:,'approved_date'] = [parse(date) if isinstance(date, str) else date for date in df['approved_date']]

# Convert dashboard savings from strings to floats
df.loc[:,'current_base_and_excercised_options_value'] = [re.sub(r'[$,]', '', str(saving)) if isinstance(saving, str) else saving for saving in df['current_base_and_excercised_options_value']]

# Get consistent counts
count_valid_date = df[df['approved_date'].notnull() & (df['deleted_date'] > df['approved_date'])].shape[0]
count_valid_savings = 0
for _, row in df.iterrows():
    if row['approved_date'] is None:
        continue
    try:
        # Check if savings_fpds are within 10,000 of the savings_dashboard
        if (abs(float(row['current_base_and_excercised_options_value']) - float(row['savings'])) < 100000):
            count_valid_savings += 1
    except Exception as e:
        print(f"Error in savings validation: {e}")
        continue
print(f"Number of contracts with valid dates: {count_valid_date}/{len(df)} = {count_valid_date/len(df):.2%}")
print(f"Number of contracts with valid savings: {count_valid_savings}/{len(df)} = {count_valid_savings/len(df):.2%}")

# Plot two-line line-plot of dashboard savings vs FPDS savings
import matplotlib.pyplot as plt
import seaborn as sns
df.loc[:,'current_base_and_excercised_options_value'] = df['current_base_and_excercised_options_value'].astype(float)
df.loc[:,'savings'] = df['savings'].astype(float)
# Trim outliers
df = df[(df['savings'] < 100000) & (df['current_base_and_excercised_options_value'] < 100000)]
def plot_savings(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='deleted_date', y='savings', label='Dashboard Savings')
    sns.lineplot(data=df, x='deleted_date', y='current_base_and_excercised_options_value', label='FPDS Savings')
    plt.title('Savings Comparison')
    plt.xlabel('Date')
    plt.ylabel('Savings')
    plt.ylim(0, max(max(df['savings']), max(df['current_base_and_excercised_options_value'])) * 1.1)
    plt.legend()
    plt.show()
# plot_savings(df)

# ---------------------------------
# Stage 2b: Qualitative Metric Computation
# ---------------------------------
# 1. Completeness metrics

# for each column in df, check if it is not null and count the number of non-null values
def completeness_metrics(df: pd.DataFrame) -> Dict[str, float]:
    completeness = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            completeness[col] = df[col].notnull().sum() / len(df)
        else:
            completeness[col] = df[col].count() / len(df)
    return completeness
# Get completeness metrics
completeness = completeness_metrics(df)
# Print completeness metrics
for col, value in completeness.items():
    print(f"Completeness of {col}: {value:.2%}")
# Plot histogram of completeness metrics. Bins are number of columns that are 10-20% complete for example
def plot_completeness(completeness: Dict[str, float]):
    plt.figure(figsize=(12, 6))
    plt.hist(list(completeness.values()), bins=20)
    plt.title('Completeness Metrics')
    plt.xlabel('Completeness')
    plt.ylabel('Number of Columns')
    plt.show()
# plot_completeness(completeness)

# Select descriptive columns
def select_descriptive_columns(df: pd.DataFrame) -> List[str]:
    # Select columns that are descriptive
    descriptive_columns = []
    for col in df.columns:
        if df[col].dtype == 'object' and len(df[col].unique()) < 10:
            descriptive_columns.append(col)
    return descriptive_columns
descriptive_columns = select_descriptive_columns(df)
# Create a list of average length of each column, not counting null values
def average_length_of_columns(df: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
    avg_length = {}
    for col in columns:
        avg_length[col] = df[col].str.len().mean()
    return avg_length
# Get average length of descriptive columns
avg_length = average_length_of_columns(df, descriptive_columns)
# Replace nan values with 0
for col in avg_length.keys():
    if pd.isna(avg_length[col]):
        avg_length[col] = 0
# Plot a histogram of average length of descriptive columns
def plot_average_length(avg_length: Dict[str, float]):
    plt.figure(figsize=(12, 6))
    plt.hist(list(avg_length.values()), bins=10)
    plt.title('Average Length of Descriptive Columns')
    plt.xlabel('Average Length')
    plt.ylabel('Number of Columns')
    plt.show()
# plot_average_length(avg_length)

# Use topic modeling to get the most common topics per column. Select the five columns with longest length on average
def topic_modeling(df: pd.DataFrame, columns: List[str]) -> Dict[str, List[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    topics = {}
    for col in columns:
        if df[col].dtype == 'object':
            if len(df[col].dropna()) == 0:
                continue
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform(df[col].dropna())
            nmf = NMF(n_components=5, random_state=1)
            nmf.fit(tfidf)
            feature_names = vectorizer.get_feature_names_out()
            topics[col] = []
            for topic_idx, topic in enumerate(nmf.components_):
                topic_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
                topics[col].append(topic_words)
    return topics
# Get topics for descriptive columns
top_descriptive_columns = sorted(avg_length, key=avg_length.get, reverse=True)[:5]
topics = topic_modeling(df, top_descriptive_columns)
# Print topics
for col, topic_list in topics.items():
    print(f"Topics for {col}:")
    for i, topic in enumerate(topic_list):
        print(f"  Topic {i+1}: {', '.join(topic)}")