from scipy.stats import chisquare
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
df = pd.read_csv(CONTRACTS_CSV_PATH, encoding='utf-8')
# Select last 30 days
df['deleted_date'] = pd.to_datetime(df['deleted_date'])
df = df[df['deleted_date'] > (datetime.now() - relativedelta(days=30))]

# Step 1: Merge DOGE data with state codes
geo = pd.read_csv(OUTPUT_CSV_PATH)
geo = geo[['piid', 'principal_place_of_performance_state_code']]
df = df.merge(geo, on='piid', how='left')
df.rename(columns={'principal_place_of_performance_state_code': 'state'}, inplace=True)

# Step 2: Filter out rows with missing state codes or agencies
doge_to_wiki_dept_map = {
    'Department of State': 'State',
    'Department of Treasury': 'Treasury',
    'Department of the Interior': 'Interior',
    'Department of Agriculture': 'Agriculture',
    'Department of Justice': 'Justice',
    'Department of Commerce': 'Commerce',
    'Department of Labor': 'Labor',
    'Department of Defense': 'Defense',
    'Department of Health and Human Services': 'Health and Human Services',
    'Housing and Urban Development': 'Housing and Urban Development',
    'Department of Transportation': 'Transportation',
    'Department of Energy': 'Energy',
    'Department of Education': 'Education',
    'Department of Veterans Affairs': 'Veterans Affairs',
    'Department of Homeland Security': 'Homeland Security',
    
    # Optional mappings (not in Wikipedia list, will be dropped or treated separately)
    'Environmental Protection Agency': None,
    'General Services Administration': None,
    'USAID': None,
    'CORPORATION FOR NATIONAL AND COMMUNITY SERVICE': None,
    'Corporation for National and Community Service': None,
    'Securities and Exchange Commission': None,
    'National Aeronautics and Space Administration': None,
    'Social Security Administration': None,
    'Office of Personnel Management': None,
    'Commodity Futures Trading Commission': None,
    'Federal Communications Commission': None,
    'UNITED STATES TRADE AND DEVELOPMENT AGENCY': None,
    'National Science Foundation': None,
    'Federal Mediation and Conciliation Service': None,
    'AmeriCorps': None,
    'Institute of Museum And Library Services': None,
    'Federal Trade Commission': None,
    'International Assistance Programs': None,
    'Smithsonian Institution': None,
    'Nuclear Regulatory Commission': None,
    'Small Business Administration': None,
    'GOVERNMENT ACCOUNTABILITY OFFICE': None
}
# Map DOGE departments to Wikipedia departments
df['agency'] = df['agency'].map(doge_to_wiki_dept_map)
# Drop rows with missing state codes or agencies
df = df.dropna(subset=['state', 'agency'])

# Step 3: Load baseline data
df_state = pd.read_csv('data/state_spending_baseline.csv')
df_dept = pd.read_csv('data/department_spending_baseline.csv')

# Step 4: Aggregate spending by state and department
observed_dept = df.groupby('agency')['savings'].sum().dropna()
observed_state = df.groupby('state')['savings'].sum().dropna()
# Make sure observed_state and df_state share the same states
observed_state = observed_state[observed_state.index.isin(df_state['state'])]

# Step 4.1: Normalize all spending data
df_dept['baseline_spend'] = df_dept['baseline_spend'] / df_dept['baseline_spend'].sum()
df_state['baseline_spend'] = df_state['baseline_spend'] / df_state['baseline_spend'].sum()
observed_dept = observed_dept / observed_dept.sum()
observed_state = observed_state / observed_state.sum()

# Step 5: Align baseline data with observed data
def chi_test(baseline_df, observed_df, baseline_col):
    # Filter baseline data to only include departments present in observed data
    baseline_df = baseline_df[baseline_df[baseline_col].isin(observed_df.index)]
    # Align the order of the baseline data with the observed data
    baseline_df = baseline_df.set_index(baseline_col).loc[observed_df.index]

    # Ensure the index is sorted
    baseline_df['expected'] = baseline_df['baseline_spend'] / baseline_df['baseline_spend'].sum() * observed_df.sum()

    # Chi-square test
    chi_stat, p_val = chisquare(f_obs=observed_df, f_exp=baseline_df['expected'])

    # Combine into one DataFrame for reporting
    comparison = pd.DataFrame({
        'observed_savings': observed_df,
        'expected_savings': baseline_df['expected'],
        'baseline_spend': baseline_df['baseline_spend']
    })
    print(len(observed_df), len(baseline_df))

    comparison['residual'] = comparison['observed_savings'] - comparison['expected_savings']
    comparison['z_score'] = (comparison['residual'] / comparison['expected_savings'].std())

    # Output
    print(f"Chi-square statistic: {chi_stat:.2f}, p-value: {p_val:.4f}")
    print(comparison.sort_values(by='z_score', ascending=False))

chi_test(df_dept, observed_dept, 'department')
chi_test(df_state, observed_state, 'state')

# Combine df_dept and observed_dept for plotting
dept = df_dept.merge(observed_dept, how='inner', left_on='department', right_index=True)
state = df_state.merge(observed_state, how='inner', left_on='state', right_index=True)

# Plot using just matplotlib
import matplotlib.pyplot as plt

def plot_spending(df, index, title):
    plt.figure(figsize=(12, 6))
    plt.bar(df[index], df["savings"], color='blue', alpha=0.5)
    plt.bar(df[index], df["baseline_spend"], color='orange', alpha=0.5)
    plt.title(title)
    plt.xlabel(index)
    plt.ylabel("Percentage of Total Spending")
    plt.ylim(0, 0.6)
    plt.legend(['Observed', 'Expected'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
# plot_spending(dept, 'department', 'Observed vs Expected Spending for Departments')  
# plot_spending(state, 'state', 'Observed vs Expected Spending for States')