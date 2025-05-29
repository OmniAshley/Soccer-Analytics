#----------------------------------------------------------#
# Name: Soccer Analytics
# Purpose: To create a complex prediction model for soccer
#          matches with high levels of accuracy.
# Author: Ashley Beebakee (https://github.com/OmniAshley)
# Date Created: 29/05/2025
# Last Updated: 29/05/2025
# Python Version: 3.10.6
#----------------------------------------------------------#
# Methodology
# Step 1: Define the Prediction Goal
# Label: Multi-class classification
# 0 = Away Win
# 1 = Draw
# 2 = Home Win
#
# Step 2: Collect Historical and Real-Time Data
# Pipelines
# A. Historical Data = for model training & validation 
#    - Go to https://www.football-data.co.uk/mmz4281/{}/E0.csv
#      where {} is the season, i.e. 2324
#    - Load Premier League 2022/23 match results
# B. Real-Time Data = for live predictions
#    - Go to https://www.api-football.com
#    - Set up free-tier API for real-time data
#
# Step 3: Preprocess & Engineer Features
#
#
# Required libraries: pandas, requests
# Acronyms: FTHG (Full Time Home Goals), FTAG(Full Time Away
#           Goals), FTR (Full Time Result(H, D, A))
# 
#----------------------------------------------------------#
# Actions
# 1. Extend Historical Data from 2020-2024 to 1999-2024
#
#
#
#----------------------------------------------------------#

import pandas as pd
import requests

###--- A. Historical Data Collection ---###
# Define season codes and URLs
season_codes = ["2021", "2122", "2223", "2324"]
base_url = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"

dfs = []

for code in season_codes:
    url = base_url.format(code)
    df = pd.read_csv(url)
    df['Season'] = code  # Add season indicator
    dfs.append(df)

# Combine all seasons into one DataFrame
all_matches = pd.concat(dfs, ignore_index=True)

# Rename for consistency
all_matches = all_matches.rename(columns={
    'HomeTeam': 'Home',
    'AwayTeam': 'Away',
    'FTHG': 'HomeGoals',
    'FTAG': 'AwayGoals',
    'FTR': 'ResultCode'
})

# Drop rows with missing results
all_matches = all_matches.dropna(subset=['HomeGoals', 'AwayGoals', 'ResultCode'])

# Map result to numerical label
result_map = {'A': 0, 'D': 1, 'H': 2}
all_matches['Result'] = all_matches['ResultCode'].map(result_map)

# Convert date
all_matches['Date'] = pd.to_datetime(all_matches['Date'], dayfirst=True, errors='coerce')

# Save Excel spreadsheet for seasons 2020 - 2024 for future use
#all_matches.to_csv("epl_matches_2020_2024.csv", index=False)

# Preview
print(all_matches[['Date', 'Home', 'Away', 'HomeGoals', 'AwayGoals', 'Result']].tail())

"""
###--- B. Real Time Data Collection (API-based) ---###
API_KEY = "6bde5cdfd4c79fe2f2bf02dab29c537d"
headers = {'X-RapidAPI-Key': API_KEY}

# Example: get next 10 Premier League fixtures
url = "https://v3.football.api-sports.io/fixtures?league=5&season=2022"
response = requests.get(url, headers=headers)
data = response.json()
print(data)
for match in data['response']:
    home = match['teams']['home']['name']
    away = match['teams']['away']['name']
    date = match['fixture']['date']
    print(f"{date} | {home} vs {away}")
"""
