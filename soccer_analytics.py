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
# Required libraries: pandas, requests, seaborn, scikit-learn
# Acronyms: FTHG (Full Time Home Goals), FTAG(Full Time Away
#           Goals), FTR (Full Time Result(H, D, A)),
#           GF (Goals For), GA (Goals Against)
# 
#----------------------------------------------------------#
# Actions
# 1. Extend Historical Data from 2020-2024 to 1999-2024
#
#
#
#----------------------------------------------------------#

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import requests

def calculate_team_stats(df, team_col='Home', opponent_col='Away', goals_for_col='HomeGoals', goals_against_col='AwayGoals'):
    stats = []

    teams = df[team_col].unique()

    for team in teams:
        team_df = df[(df[team_col] == team) | (df[opponent_col] == team)].copy()
        team_df = team_df.sort_values('Date')

        # Record historical stats
        team_stats = {
            'Team': [],
            'Date': [],
            'GF_avg': [],
            'GA_avg': [],
            'WinRate': [],
            'RecentPoints': [],
        }

        past_matches = []

        for idx, row in team_df.iterrows():
            is_home = row[team_col] == team
            gf = row[goals_for_col] if is_home else row[goals_against_col]
            ga = row[goals_against_col] if is_home else row[goals_for_col]
            result = row['Result']

            # Compute past stats
            if len(past_matches) >= 5:
                past = pd.DataFrame(past_matches[-5:])
                gf_avg = past['GF'].mean()
                ga_avg = past['GA'].mean()
                win_rate = (past['Points'] == 3).mean()
                total_pts = past['Points'].sum()
            else:
                gf_avg = ga_avg = win_rate = total_pts = None

            team_stats['Team'].append(team)
            team_stats['Date'].append(row['Date'])
            team_stats['GF_avg'].append(gf_avg)
            team_stats['GA_avg'].append(ga_avg)
            team_stats['WinRate'].append(win_rate)
            team_stats['RecentPoints'].append(total_pts)

            # Update history
            points = 3 if (result == 2 and is_home) or (result == 0 and not is_home) else 1 if result == 1 else 0
            past_matches.append({'GF': gf, 'GA': ga, 'Points': points})

        team_stats_df = pd.DataFrame(team_stats)
        stats.append(team_stats_df)

    return pd.concat(stats, ignore_index=True)

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
#print(all_matches[['Date', 'Home', 'Away', 'HomeGoals', 'AwayGoals', 'Result']].tail())

# Compute rolling features for both home and away teams
home_stats = calculate_team_stats(all_matches, 'Home', 'Away', 'HomeGoals', 'AwayGoals')
away_stats = calculate_team_stats(all_matches, 'Away', 'Home', 'AwayGoals', 'HomeGoals')

# Rename columns
home_stats = home_stats.rename(columns={
    'GF_avg': 'Home_GF_avg', 'GA_avg': 'Home_GA_avg',
    'WinRate': 'Home_WinRate', 'RecentPoints': 'Home_RecentPoints'
})

away_stats = away_stats.rename(columns={
    'GF_avg': 'Away_GF_avg', 'GA_avg': 'Away_GA_avg',
    'WinRate': 'Away_WinRate', 'RecentPoints': 'Away_RecentPoints'
})

# Merge with main dataframe
features_df = all_matches.merge(home_stats, left_on=['Home', 'Date'], right_on=['Team', 'Date'], how='left')
features_df = features_df.merge(away_stats, left_on=['Away', 'Date'], right_on=['Team', 'Date'], how='left')

# Drop extra columns
features_df.drop(columns=['Team_x', 'Team_y'], inplace=True)

# Preview
print(features_df[[
    'Date', 'Home', 'Away',
    'Home_GF_avg', 'Home_GA_avg', 'Home_WinRate', 'Home_RecentPoints',
    'Away_GF_avg', 'Away_GA_avg', 'Away_WinRate', 'Away_RecentPoints',
    'Result'
]].tail())

# Select numeric features
corr_matrix = features_df[[
    'Home_GF_avg', 'Home_GA_avg', 'Home_WinRate', 'Home_RecentPoints',
    'Away_GF_avg', 'Away_GA_avg', 'Away_WinRate', 'Away_RecentPoints',
    'Result'
]].corr()

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix (Features vs Result)")
#plt.show()

features = [
    'Home_GF_avg', 'Home_GA_avg', 'Home_WinRate', 'Home_RecentPoints',
    'Away_GF_avg', 'Away_GA_avg', 'Away_WinRate', 'Away_RecentPoints'
]

X = features_df[features]
y = features_df['Result']

X = X.dropna()
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100102, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=100102)
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

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
