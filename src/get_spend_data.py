"""This script is a one-time run to extract spending distribution data from Wikipedia and USASpending.gov."""

import pandas as pd
import requests
import re
from bs4 import BeautifulSoup

# URL of the Wikipedia page
url = "https://en.wikipedia.org/wiki/List_of_U.S._state_budgets"

# Fetch and parse the HTML
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find the first table (contains state budget data)
table = soup.find("table", class_="wikitable")

# Parse table rows
data = []
for row in table.find_all("tr")[1:]:  # skip header
    cols = row.find_all(["td", "th"])
    if len(cols) < 2:
        continue
    state = cols[0].get_text(strip=True)
    budget_str = cols[1].get_text(strip=True).replace("$", "").replace(",", "").split()[0]
    try:
        budget = float(budget_str)
        data.append({"state": state, "baseline_spend": budget})
    except ValueError:
        continue  # skip rows with bad formatting

# Create DataFrame
df = pd.DataFrame(data)

# Map state names to abbreviations
state_name_to_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}
df['state'] = df['state'].replace(state_name_to_abbrev)
# Save to CSV
df.to_csv("data/state_spending_baseline.csv", index=False)

# URL of the Wikipedia page
url = "https://en.wikipedia.org/wiki/United_States_federal_executive_departments"

# Fetch and parse the HTML
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find the table containing department data
table = soup.find("table", class_="wikitable")

# Parse table rows
data = []
for row in table.find_all("tr")[1:]:  # skip header
    cols = row.find_all(["td", "th"])
    if len(cols) < 6:
        continue
    department = cols[0].get_text(strip=True)
    budget_str = cols[5].get_text(strip=True)
    # Extract numeric value from budget string
    match = re.search(r"\$([\d\.]+)\s*(billion|trillion)", budget_str, re.IGNORECASE)
    if match:
        amount = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "billion":
            budget = amount * 1e9
        elif unit == "trillion":
            budget = amount * 1e12
        else:
            continue
        data.append({"department": department, "baseline_spend": budget})

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("data/department_spending_baseline.csv", index=False)