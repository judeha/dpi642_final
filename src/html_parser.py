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
df = pd.read_csv(CONTRACTS_CSV_PATH, encoding='utf-8')
# Select last 30 days
df['deleted_date'] = pd.to_datetime(df['deleted_date'])
df = df[df['deleted_date'] > (datetime.now() - relativedelta(days=30))]

# ---------------------------------
# Stage 1a: Data Aggregation
# ---------------------------------
def num_completeness_fpds(df):
    # Get number of contracts with a supporting FPDS document
    num_incomplete = len(df[df['fpds_status']=="Unavailable"])
    return len(df) - num_incomplete

def num_completeness_basic(df):
    # Get number of contracts with all basic fields filled out
    num_incomplete = len(df[(df.fpds_status == "Unavailable") | (df.savings == 0)])
    return len(df) - num_incomplete

ncf = num_completeness_fpds(df)
ncb = num_completeness_basic(df)
print(f"Number of contracts with FPDS documents: {ncf}/{len(df)} = {ncf/len(df):.2%}")
print(f"Number of contracts with all basic fields filled out: {ncb}/{len(df)} = {ncb/len(df):.2%}")

# ---------------------------------
# Stage 1b: HTML Parsing
# ---------------------------------
def grab(attr_id, soup: BeautifulSoup) -> str:
    # Helper function to extract text from a tag with a specific ID
    tag = soup.find(id=attr_id)
    return tag.get_text(strip=True) if tag else None

def get_html(url: str):
    # Fetch HTML content from a given URL and return a soup object"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        return soup
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""
    
def parse_html_1_2(soup: BeautifulSoup) -> Dict[str, Any]:
    # Get the temporal and personnel fields from a soup object and return them as a dictionary
    fields = {
        "prepared_date": grab("displayPreparedDate", soup),
        "last_modified_date": grab("displayLastModifiedDate", soup),
        "approved_date": grab("displayApprovedDate", soup),
        "closed_date": grab("lblDisplayClosedDate", soup),  # label exists even when blank
        "prepared_by": grab("displayPreparedBy", soup),
        "last_modified_by": grab("displayLastModifiedBy", soup),
        "closed_by": grab("displayClosedBy", soup),
        "approved_by": grab("displayApprovedBy", soup)
    }
    return fields

def parse_html_3(soup: BeautifulSoup) -> Dict[str, Any]:
    # ---------- Category 3 : all <input>/<select> with `title` ----------
    misc = {}
    for tag in soup.select("input[title], select[title]"):
        col = tag["title"].strip().lower().replace(" ", "_")
        if tag.name == "input":
            misc[col] = tag.get("value", "").strip()
        else:  # <select>
            sel = tag.find("option", selected=True) or tag.find("option", attrs={"disabled": True})
            misc[col] = sel.text.strip() if sel else ""
    return misc



# Only parse contracts with a supporting FPDS document
df = df[df['fpds_status'] != "Unavailable"]
# Subsample 500 contracts for testing
df = df.sample(500, random_state=42)
print(f"Parsing {len(df)} HTML files")
rows = []
for _, contract in df.iterrows():
    try:
        dates_and_users = parse_html_1_2(get_html(contract["fpds_link"]))
        misc = parse_html_3(get_html(contract["fpds_link"]))
        rows.append({**contract.to_dict(), **dates_and_users, **misc})
        print(".", end="", flush=True)
    except Exception as e:
        continue
        
out = pd.DataFrame(rows)
out.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Parsed FPDS dataframe ({out.shape}) and saved to {OUTPUT_CSV_PATH}")
