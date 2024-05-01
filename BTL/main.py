# Primary libraries
from time import time
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
# # Neural Networks
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers import Input
# from keras.models import Model
# from keras.utils import np_utils
# Measures
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# Step 1: Data Analysis
# Load data
with sqlite3.connect("dataset/database.sqlite") as con:
    matches = pd.read_sql_query("SELECT * from Match", con)
    team_attributes = pd.read_sql_query("SELECT distinct * from Team_Attributes", con)
    player = pd.read_sql_query("SELECT * from Player", con)
    player_attributes = pd.read_sql_query("SELECT * from Player_Attributes", con)

# Matches
# Cleaning the match data and defining some methods for the data extraction and the labels
''' Derives a label for a given match. '''


def get_match_outcome(match):
    # Define variables
    home_goals = match['home_team_goal']
    away_goals = match['away_team_goal']

    outcome = pd.DataFrame()  # Create a new dataframe: outcome
    outcome.loc[0, 'match_api_id'] = match['match_api_id']  # Insert match_api_id into outcome

    # Identify match outcome
    if home_goals > away_goals:
        outcome.loc[0, 'outcome'] = "Win"
    if home_goals == away_goals:
        outcome.loc[0, 'outcome'] = "Draw"
    if home_goals < away_goals:
        outcome.loc[0, 'outcome'] = "Defeat"

    # Return outcome
    return outcome.loc[0]


''' Get the last x matches of a given team. '''


def get_last_matches(matches, date, team, x=10):
    # Filter team matches from matches
    team_matches = matches[(matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)]

    # Filter x last matches from team matches
    last_matches = team_matches[team_matches.date < date].sort_values(by='date', ascending=False).iloc[0:x, :]

    # Return last matches
    return last_matches


''' Get the last team stats of a given team. '''


def get_last_team_stats(team_id, date, teams_stats):
    # Filter team stats
    all_team_stats = teams_stats[teams_stats['team_api_id'] == team_id]

    # Filter last stats from team
    last_team_stats = all_team_stats[all_team_stats.date < date].sort_values(by='date', ascending=False)
    if last_team_stats.empty:
        last_team_stats = all_team_stats[all_team_stats.date > date].sort_values(by='date', ascending=True)

    # Return last matches
    return last_team_stats.iloc[0:1, :]


''' Get the last x matches of two given teams. '''


def get_last_matches_against_eachother(matches, date, home_team, away_team, x=10):
    # Find matches of both teams
    home_matches = matches[(matches['home_team_api_id'] == home_team) & (matches['away_team_api_id'] == away_team)]
    away_matches = matches[(matches['home_team_api_id'] == away_team) & (matches['away_team_api_id'] == home_team)]
    total_matches = pd.concat([home_matches, away_matches])

    # Get last x matches
    try:
        last_matches = total_matches[total_matches.date < date].sort_values(by='date', ascending=False).iloc[0:x, :]
    except:
        last_matches = total_matches[total_matches.date < date].sort_values(by='date', ascending=False).iloc[
                       0:total_matches.shape[0], :]

        # Check for error in data
        if (last_matches.shape[0] > x):
            print("Error in obtaining matches")

    # Return data
    return last_matches


''' Get the goals[home & away] of a specfic team from a set of matches. '''


def get_goals(matches, team):
    home_goals = int(matches.home_team_goal[matches.home_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.away_team_api_id == team].sum())

    total_goals = home_goals + away_goals

    return total_goals


''' Get the goals[home & away] conceided of a specfic team from a set of matches. '''


def get_goals_conceided(matches, team):
    home_goals = int(matches.home_team_goal[matches.away_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.home_team_api_id == team].sum())

    total_goals = home_goals + away_goals

    return total_goals


''' Get the number of wins of a specfic team from a set of matches. '''


def get_wins(matches, team):
    # Find home and away wins
    home_wins = int(matches.home_team_goal[
                        (matches.home_team_api_id == team) & (matches.home_team_goal > matches.away_team_goal)].count())
    away_wins = int(matches.away_team_goal[
                        (matches.away_team_api_id == team) & (matches.away_team_goal > matches.home_team_goal)].count())

    total_wins = home_wins + away_wins

    return total_wins


''' Create match specific features for a given match. '''


def get_match_features(match, matches, teams_stats, x=10):
    # Define variables
    date = match.date
    home_team = match.home_team_api_id
    away_team = match.away_team_api_id

    # Gets home and away team_stats
    home_team_stats = get_last_team_stats(home_team, date, teams_stats);
    away_team_stats = get_last_team_stats(away_team, date, teams_stats);

    # Get last x matches of home and away team
    matches_home_team = get_last_matches(matches, date, home_team, x=5)
    matches_away_team = get_last_matches(matches, date, away_team, x=5)

    # Get last x matches of both teams against each other
    last_matches_against = get_last_matches_against_eachother(matches, date, home_team, away_team, x=3)

    # Create goal variables
    home_goals = get_goals(matches_home_team, home_team)
    away_goals = get_goals(matches_away_team, away_team)
    home_goals_conceided = get_goals_conceided(matches_home_team, home_team)
    away_goals_conceided = get_goals_conceided(matches_away_team, away_team)

    # Define result data frame
    result = pd.DataFrame()

    # Define ID features
    result.loc[0, 'match_api_id'] = match.match_api_id
    result.loc[0, 'league_id'] = match.league_id

    # Create match features and team stats
    if not home_team_stats.empty:
        result.loc[0, 'home_team_buildUpPlaySpeed'] = home_team_stats['buildUpPlaySpeed'].values[0]
        result.loc[0, 'home_team_buildUpPlayPassing'] = home_team_stats['buildUpPlayPassing'].values[0]
        result.loc[0, 'home_team_chanceCreationPassing'] = home_team_stats['chanceCreationPassing'].values[0]
        result.loc[0, 'home_team_chanceCreationCrossing'] = home_team_stats['chanceCreationCrossing'].values[0]
        result.loc[0, 'home_team_chanceCreationShooting'] = home_team_stats['chanceCreationShooting'].values[0]
        result.loc[0, 'home_team_defencePressure'] = home_team_stats['defencePressure'].values[0]
        result.loc[0, 'home_team_defenceAggression'] = home_team_stats['defenceAggression'].values[0]
        result.loc[0, 'home_team_defenceTeamWidth'] = home_team_stats['defenceTeamWidth'].values[0]
        result.loc[0, 'home_team_avg_shots'] = home_team_stats['avg_shots'].values[0]
        result.loc[0, 'home_team_avg_corners'] = home_team_stats['avg_corners'].values[0]
        result.loc[0, 'home_team_avg_crosses'] = away_team_stats['avg_crosses'].values[0]

    if (not away_team_stats.empty):
        result.loc[0, 'away_team_buildUpPlaySpeed'] = away_team_stats['buildUpPlaySpeed'].values[0]
        result.loc[0, 'away_team_buildUpPlayPassing'] = away_team_stats['buildUpPlayPassing'].values[0]
        result.loc[0, 'away_team_chanceCreationPassing'] = away_team_stats['chanceCreationPassing'].values[0]
        result.loc[0, 'away_team_chanceCreationCrossing'] = away_team_stats['chanceCreationCrossing'].values[0]
        result.loc[0, 'away_team_chanceCreationShooting'] = away_team_stats['chanceCreationShooting'].values[0]
        result.loc[0, 'away_team_defencePressure'] = away_team_stats['defencePressure'].values[0]
        result.loc[0, 'away_team_defenceAggression'] = away_team_stats['defenceAggression'].values[0]
        result.loc[0, 'away_team_defenceTeamWidth'] = away_team_stats['defenceTeamWidth'].values[0]
        result.loc[0, 'away_team_avg_shots'] = away_team_stats['avg_shots'].values[0]
        result.loc[0, 'away_team_avg_corners'] = away_team_stats['avg_corners'].values[0]
        result.loc[0, 'away_team_avg_crosses'] = away_team_stats['avg_crosses'].values[0]

    result.loc[0, 'home_team_goals_difference'] = home_goals - home_goals_conceided
    result.loc[0, 'away_team_goals_difference'] = away_goals - away_goals_conceided
    result.loc[0, 'games_won_home_team'] = get_wins(matches_home_team, home_team)
    result.loc[0, 'games_won_away_team'] = get_wins(matches_away_team, away_team)
    result.loc[0, 'games_against_won'] = get_wins(last_matches_against, home_team)
    result.loc[0, 'games_against_lost'] = get_wins(last_matches_against, away_team)
    result.loc[0, 'B365H'] = match.B365H
    result.loc[0, 'B365D'] = match.B365D
    result.loc[0, 'B365A'] = match.B365A

    # Return match features
    return result.loc[0]


''' Create and aggregate features and labels for all matches. '''


def get_features(matches, teams_stats, fifa, x=10, get_overall=False):
    # Get fifa stats features
    fifa_stats = get_overall_fifa_rankings(fifa, get_overall)

    # Get match features for all matches
    match_stats = matches.apply(lambda i: get_match_features(i, matches, teams_stats, x=10), axis=1)

    # Create dummies for league ID feature
    dummies = pd.get_dummies(match_stats['league_id']).rename(columns=lambda x: 'League_' + str(x))
    match_stats = pd.concat([match_stats, dummies], axis=1)
    match_stats.drop(['league_id'], inplace=True, axis=1)

    # Create match outcomes
    outcomes = matches.apply(get_match_outcome, axis=1)

    # Merges features and outcomes into one frame
    features = pd.merge(match_stats, fifa_stats, on='match_api_id', how='left')
    features = pd.merge(features, outcomes, on='match_api_id', how='left')

    # Drop NA values
    features.dropna(inplace=True)

    # Return preprocessed data
    return features


def get_overall_fifa_rankings(fifa, get_overall=False):
    ''' Get overall fifa rankings from fifa data. '''

    temp_data = fifa

    # Check if only overall player stats are desired
    if get_overall == True:

        # Get overall stats
        data = temp_data.loc[:, (fifa.columns.str.contains('overall_rating'))]
        data.loc[:, 'match_api_id'] = temp_data.loc[:, 'match_api_id']
    else:

        # Get all stats except for stat date
        cols = fifa.loc[:, (fifa.columns.str.contains('date_stat'))]
        temp_data = fifa.drop(cols.columns, axis=1)
        data = temp_data

    # Return data
    return data


viable_matches = matches
viable_matches.describe()

viable_matches = matches.sample(n=5000)
b365 = viable_matches.dropna(subset=['B365H', 'B365D', 'B365A'], inplace=False)
b365.drop(['BWH', 'BWD', 'BWA',
           'IWH', 'IWD', 'IWA',
           'LBH', 'LBD', 'LBA',
           'PSH', 'PSD', 'PSA',
           'WHH', 'WHD', 'WHA',
           'SJH', 'SJD', 'SJA',
           'VCH', 'VCD', 'VCA',
           'GBH', 'GBD', 'GBA',
           'BSH', 'BSD', 'BSA'], inplace=True, axis=1)

bw = viable_matches.dropna(subset=['BWH', 'BWD', 'BWA'], inplace=False)
bw.drop(['B365H', 'B365D', 'B365A',
         'IWH', 'IWD', 'IWA',
         'LBH', 'LBD', 'LBA',
         'PSH', 'PSD', 'PSA',
         'WHH', 'WHD', 'WHA',
         'SJH', 'SJD', 'SJA',
         'VCH', 'VCD', 'VCA',
         'GBH', 'GBD', 'GBA',
         'BSH', 'BSD', 'BSA'], inplace=True, axis=1)

iw = viable_matches.dropna(subset=['IWH', 'IWD', 'IWA'], inplace=False)
iw.drop(['B365H', 'B365D', 'B365A',
         'BWH', 'BWD', 'BWA',
         'LBH', 'LBD', 'LBA',
         'PSH', 'PSD', 'PSA',
         'WHH', 'WHD', 'WHA',
         'SJH', 'SJD', 'SJA',
         'VCH', 'VCD', 'VCA',
         'GBH', 'GBD', 'GBA',
         'BSH', 'BSD', 'BSA'], inplace=True, axis=1)

lb = viable_matches.dropna(subset=['LBH', 'LBD', 'LBA'], inplace=False)
lb.drop(['B365H', 'B365D', 'B365A',
         'BWH', 'BWD', 'BWA',
         'IWH', 'IWD', 'IWA',
         'PSH', 'PSD', 'PSA',
         'WHH', 'WHD', 'WHA',
         'SJH', 'SJD', 'SJA',
         'VCH', 'VCD', 'VCA',
         'GBH', 'GBD', 'GBA',
         'BSH', 'BSD', 'BSA'], inplace=True, axis=1)

ps = viable_matches.dropna(subset=['PSH', 'PSD', 'PSA'], inplace=False)
ps.drop(['B365H', 'B365D', 'B365A',
         'BWH', 'BWD', 'BWA',
         'IWH', 'IWD', 'IWA',
         'LBH', 'LBD', 'LBA',
         'WHH', 'WHD', 'WHA',
         'SJH', 'SJD', 'SJA',
         'VCH', 'VCD', 'VCA',
         'GBH', 'GBD', 'GBA',
         'BSH', 'BSD', 'BSA'], inplace=True, axis=1)

wh = viable_matches.dropna(subset=['WHH', 'WHD', 'WHA'], inplace=False)
wh.drop(['B365H', 'B365D', 'B365A',
         'BWH', 'BWD', 'BWA',
         'IWH', 'IWD', 'IWA',
         'LBH', 'LBD', 'LBA',
         'PSH', 'PSD', 'PSA',
         'SJH', 'SJD', 'SJA',
         'VCH', 'VCD', 'VCA',
         'GBH', 'GBD', 'GBA',
         'BSH', 'BSD', 'BSA'], inplace=True, axis=1)

sj = viable_matches.dropna(subset=['SJH', 'SJD', 'SJA'], inplace=False)
sj.drop(['B365H', 'B365D', 'B365A',
         'BWH', 'BWD', 'BWA',
         'IWH', 'IWD', 'IWA',
         'LBH', 'LBD', 'LBA',
         'PSH', 'PSD', 'PSA',
         'WHH', 'WHD', 'WHA',
         'VCH', 'VCD', 'VCA',
         'GBH', 'GBD', 'GBA',
         'BSH', 'BSD', 'BSA'], inplace=True, axis=1)

vc = viable_matches.dropna(subset=['VCH', 'VCD', 'VCA'], inplace=False)
vc.drop(['B365H', 'B365D', 'B365A',
         'BWH', 'BWD', 'BWA',
         'IWH', 'IWD', 'IWA',
         'LBH', 'LBD', 'LBA',
         'PSH', 'PSD', 'PSA',
         'WHH', 'WHD', 'WHA',
         'SJH', 'SJD', 'SJA',
         'GBH', 'GBD', 'GBA',
         'BSH', 'BSD', 'BSA'], inplace=True, axis=1)

gb = viable_matches.dropna(subset=['GBH', 'GBD', 'GBA'], inplace=False)
gb.drop(['B365H', 'B365D', 'B365A',
         'BWH', 'BWD', 'BWA',
         'IWH', 'IWD', 'IWA',
         'LBH', 'LBD', 'LBA',
         'PSH', 'PSD', 'PSA',
         'WHH', 'WHD', 'WHA',
         'SJH', 'SJD', 'SJA',
         'VCH', 'VCD', 'VCA',
         'BSH', 'BSD', 'BSA'], inplace=True, axis=1)

bs = viable_matches.dropna(subset=['BSH', 'BSD', 'BSA'], inplace=False)
bs.drop(['B365H', 'B365D', 'B365A',
         'BWH', 'BWD', 'BWA',
         'IWH', 'IWD', 'IWA',
         'LBH', 'LBD', 'LBA',
         'PSH', 'PSD', 'PSA',
         'WHH', 'WHD', 'WHA',
         'SJH', 'SJD', 'SJA',
         'VCH', 'VCD', 'VCA',
         'GBH', 'GBD', 'GBA'], inplace=True, axis=1)

lis = [b365, bw, iw, lb, ps, wh, sj, vc, gb, bs]

viable_matches = max(lis, key=lambda datframe: datframe.shape[0])
viable_matches.describe()

# Remove some rows that do not contain any information about the position of the players for some matches.

teams_stats = team_attributes
viable_matches = viable_matches.dropna(inplace=False)

home_teams = viable_matches['home_team_api_id'].isin(teams_stats['team_api_id'].tolist())
away_teams = viable_matches['away_team_api_id'].isin(teams_stats['team_api_id'].tolist())
viable_matches = viable_matches[home_teams & away_teams]

viable_matches.describe()

# Team stats - Team attributes
teams_stats.describe()

# mean imputation for missing value
teams_stats['buildUpPlayDribbling'].hist();

build_up_play_drib_avg = teams_stats['buildUpPlayDribbling'].mean()
# mean imputation
teams_stats.loc[(teams_stats['buildUpPlayDribbling'].isnull()), 'buildUpPlayDribbling'] = build_up_play_drib_avg
# showing new values
teams_stats.loc[teams_stats['buildUpPlayDribbling'] == build_up_play_drib_avg].head()

teams_stats.loc[(teams_stats['buildUpPlayDribbling'].isnull())]

# select only continuous data
teams_stats.drop(['buildUpPlaySpeedClass', 'buildUpPlayDribblingClass', 'buildUpPlayPassingClass',
                  'buildUpPlayPositioningClass', 'chanceCreationPassingClass', 'chanceCreationCrossingClass',
                  'chanceCreationShootingClass', 'chanceCreationPositioningClass', 'defencePressureClass',
                  'defenceAggressionClass',
                  'defenceTeamWidthClass', 'defenceDefenderLineClass'], inplace=True, axis=1)

teams_stats.describe()

# Team Stats - Shots
shots_off = pd.read_csv("dataset/shotoff_detail.csv")
shots_on = pd.read_csv("dataset/shoton_detail.csv")
shots = pd.concat([shots_off[['match_id', 'team']], shots_on[['match_id', 'team']]])

total_shots = shots["team"].value_counts()
total_matches = shots.drop_duplicates(['match_id', 'team'])["team"].value_counts()

for index, n_shots in total_shots.items():
    n_matches = total_matches[index]
    avg_shots = n_shots / n_matches
    teams_stats.loc[teams_stats['team_api_id'] == index, 'avg_shots'] = avg_shots

teams_stats.describe()

teams_stats['avg_shots'].hist();

shots_avg_team_avg = teams_stats['avg_shots'].mean()
# mean imputation
teams_stats.loc[(teams_stats['avg_shots'].isnull()), 'avg_shots'] = shots_avg_team_avg
# showing new values
teams_stats.describe()

teams_stats.loc[(teams_stats['avg_shots'].isnull())]

# Team Stats - Possession
# possessions read, cleanup and merge
possessions_data = pd.read_csv("dataset/possession_detail.csv")
last_possessions = possessions_data.sort_values(['elapsed'], ascending=False).drop_duplicates(subset=['match_id'])
last_possessions = last_possessions[['match_id', 'homepos', 'awaypos']]

# get the ids of the home_team and away_team to be able to join with teams later
possessions = pd.DataFrame(columns=['team', 'possession', 'match'])
for index, row in last_possessions.iterrows():
    match = matches.loc[matches['id'] == row['match_id'], ['home_team_api_id', 'away_team_api_id']]
    if match.empty:
        continue
    hometeam = match['home_team_api_id'].values[0]
    awayteam = match['away_team_api_id'].values[0]
    possessions = possessions._append({'team': hometeam, 'possession': row['homepos'], 'match': row['match_id']},
                                      ignore_index=True)
    possessions = possessions._append({'team': awayteam, 'possession': row['awaypos'], 'match': row['match_id']},
                                      ignore_index=True)

total_possessions = possessions.groupby(by=['team'])['possession'].sum()
total_matches = possessions.drop_duplicates(['team', 'match'])["team"].value_counts()

total_possessions.to_frame().describe()
# Decide to scrap this attribute -> The poor usability of this dataset

# Team Stats - Corners
corners_data = pd.read_csv("dataset/corner_detail.csv")
corners = corners_data[['match_id', 'team']]

total_corners = corners["team"].value_counts()
total_matches = corners.drop_duplicates(['match_id', 'team'])["team"].value_counts()

for index, n_corners in total_shots.items():
    n_matches = total_matches[index]
    avg_corners = n_corners / n_matches
    teams_stats.loc[teams_stats['team_api_id'] == index, 'avg_corners'] = avg_corners

teams_stats.describe()

teams_stats['avg_corners'].hist();

corners_avg_team_avg = teams_stats['avg_corners'].mean()
# mean imputation
teams_stats.loc[(teams_stats['avg_corners'].isnull()), 'avg_corners'] = corners_avg_team_avg
# showing new values
teams_stats.describe()

teams_stats.loc[(teams_stats['avg_corners'].isnull())]

# Team Stats - Crosses
crosses_data = pd.read_csv("dataset/cross_detail.csv")

crosses = crosses_data[['match_id', 'team']]
total_crosses = crosses["team"].value_counts()
total_matches = crosses.drop_duplicates(['match_id', 'team'])["team"].value_counts()

for index, n_crosses in total_crosses.items():
    n_matches = total_matches[index]
    avg_crosses = n_crosses / n_matches
    teams_stats.loc[teams_stats['team_api_id'] == index, 'avg_crosses'] = avg_crosses

teams_stats.describe()

teams_stats['avg_crosses'].hist();

crosses_avg_team_avg = teams_stats['avg_crosses'].mean()
# mean imputation
teams_stats.loc[(teams_stats['avg_crosses'].isnull()), 'avg_crosses'] = crosses_avg_team_avg
# showing new values
teams_stats.describe()

teams_stats.loc[(teams_stats['avg_crosses'].isnull())]


# FIFA data
def get_fifa_stats(match, player_stats):
    ''' Aggregates fifa stats for a given match. '''

    # Define variables
    match_id = match.match_api_id
    date = match['date']
    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    player_stats_new = pd.DataFrame()
    names = []

    # Loop through all players
    for player in players:

        # Get player ID
        player_id = match[player]

        # Get player stats
        stats = player_stats[player_stats.player_api_id == player_id]

        # Identify current stats
        current_stats = stats[stats.date < date].sort_values(by='date', ascending=False)[:1]

        if np.isnan(player_id) == True:
            overall_rating = pd.Series(0)
        else:
            current_stats.reset_index(inplace=True, drop=True)
            overall_rating = pd.Series(current_stats.loc[0, "overall_rating"])

        # Rename stat
        name = "{}_overall_rating".format(player)
        names.append(name)

        # Aggregate stats
        player_stats_new = pd.concat([player_stats_new, overall_rating], axis=1)

    player_stats_new.columns = names
    player_stats_new['match_api_id'] = match_id

    player_stats_new.reset_index(inplace=True, drop=True)

    # Return player stats
    return player_stats_new.iloc[0]


def get_fifa_data(matches, player_stats, path=None, data_exists=False):
    ''' Gets fifa data for all matches. '''

    # Check if fifa data already exists
    if data_exists == True:
        fifa_data = pd.read_pickle(path)

    else:
        print("Collecting fifa data for each match...")
        start = time()

        # Apply get_fifa_stats for each match
        fifa_data = matches.apply(lambda x: get_fifa_stats(x, player_stats), axis=1)

        end = time()
        print("Fifa data collected in {:.1f} minutes".format((end - start) / 60))

    # Return fifa_data
    return fifa_data


fifa_data = get_fifa_data(viable_matches, player_attributes, None, data_exists=False)
fifa_data.describe()

# Joining all features
# Creates features and labels based on the provided data
viables = get_features(viable_matches, teams_stats, fifa_data, 10, False)
inputs = viables.drop('match_api_id', axis=1)
outcomes = inputs.loc[:, 'outcome']
# all features except outcomes
features = inputs.drop('outcome', axis=1)
features.iloc[:, :] = Normalizer(norm='l1').fit_transform(features)
print(features)


# Step 2: Classification & Results Interpretation
# Training and Evaluating Models
# K-Fold Cross validation.
def convert_xgb_labels(labels):
    xgb_labels = labels.copy()
    for i in range(len(xgb_labels)):
        if xgb_labels[i] == 'Win':
            xgb_labels[i] = 0
        elif xgb_labels[i] == 'Draw':
            xgb_labels[i] = 1
        else:
            xgb_labels[i] = 2
    return xgb_labels

def convert_xgb_y_predict(y_predict):
    mapping_dict = {0: 'Win', 1: 'Draw', 2: 'Defeat'}
    y_predict = np.array([mapping_dict[value] for value in y_predict], dtype=object)
    return y_predict

def train_predict(clf, data, outcomes):
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(data)))
    if clf.__class__.__name__ == 'XGBClassifier':
        xgb_outcomes = convert_xgb_labels(outcomes)
        y_predict = train_model(clf, data, xgb_outcomes)
        predict_metrics(outcomes, convert_xgb_y_predict(y_predict))
    else:
        y_predict = train_model(clf, data, outcomes)
        predict_metrics(outcomes, y_predict)


# def train_predict_nn(clf, data, outcomes):
#     le = LabelEncoder()
#     y_outcomes = le.fit_transform(outcomes)
#     y_outcomes = np_utils.to_categorical(y_outcomes)

#     y_predict = train_model_nn(clf, data, y_outcomes)

#     y_predict_reverse = [np.argmax(y, axis=None, out=None) for y in y_predict]
#     y_predict_decoded = le.inverse_transform(y_predict_reverse)
#     predict_metrics(outcomes, y_predict_decoded)

def train_model(clf, data, labels):
    kf = KFold(n_splits=5)
    predictions = []
    for train, test in kf.split(data):
        X_train, X_test = data[data.index.isin(train)], data[data.index.isin(test)]
        y_train, y_test = labels[data.index.isin(train)], labels[data.index.isin(test)]
        clf.fit(X_train, y_train)
        predictions.append(clf.predict(X_test))

    y_predict = predictions[0]
    y_predict = np.append(y_predict, predictions[1], axis=0)
    y_predict = np.append(y_predict, predictions[2], axis=0)
    y_predict = np.append(y_predict, predictions[3], axis=0)
    y_predict = np.append(y_predict, predictions[4], axis=0)

    return y_predict


# def train_model_nn(clf, data, labels):
#     kf = KFold(n_splits=5, shuffle=False)
#     predictions = []
#     for train, test in kf.split(data):
#         X_train, X_test = data[data.index.isin(train)], data[data.index.isin(test)]
#         y_train, y_test = labels[data.index.isin(train)], labels[data.index.isin(test)]
#         clf.fit(X_train, y_train, epochs=20, verbose=0)
#         predictions.append(clf.predict(X_test))

#     y_predict = predictions[0]
#     y_predict = np.append(y_predict, predictions[1], axis=0)
#     y_predict = np.append(y_predict, predictions[2], axis=0)
#     y_predict = np.append(y_predict, predictions[3], axis=0)
#     y_predict = np.append(y_predict, predictions[4], axis=0)

#     return y_predict

def predict_metrics(y_test, y_predict):
    ls = ['Win', 'Draw', 'Defeat']
    from sklearn import metrics
    # cm = metrics.confusion_matrix(y_test, y_predict, ls)
    # disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ls)
    # disp.plot(include_values=True, values_format='d')
    # plt.show()

    print(metrics.classification_report(y_test, y_predict, target_names=ls))
    print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))
    print("Recall: ", metrics.recall_score(y_test, y_predict, average='macro'))
    print("Precision: ", metrics.precision_score(y_test, y_predict, average='macro', zero_division=0))
    print("F1 Score: ", metrics.f1_score(y_test, y_predict, average='macro'))
    print("-----------------------------------")


# KNN
clf = KNeighborsClassifier(n_neighbors=90)
train_predict(clf, features, outcomes)

# Decision Tree
clf = DecisionTreeClassifier(random_state=0, criterion='entropy', splitter='random', max_depth=5)
train_predict(clf, features, outcomes)

#RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
train_predict(clf, features, outcomes)

# SVC
clf = SVC(coef0=5, kernel='poly')
train_predict(clf, features, outcomes)

# Naive Bayes
clf = GaussianNB(var_smoothing=1.1)
train_predict(clf, features, outcomes)

# # XGBoost
clf = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)
train_predict(clf, features, outcomes)

# # Neural Network
# visible = Input(shape=(features.shape[1],))
# hidden = Dense(500, activation='relu')(visible)
# output = Dense(3, activation='softmax')(hidden)

# clf = Model(inputs=visible, outputs=output)
# print(clf.summary())

# from keras import metrics
# from keras import losses
# from keras import optimizers

# clf.compile(optimizer=optimizers.Adam(),
#               loss=losses.CategoricalCrossentropy(),
#               metrics=[metrics.Precision(), metrics.Recall()])

# train_predict_nn(clf, features, outcomes)

# # Deep Neural Network
# visible = Input(shape=(features.shape[1],))
# hidden1 = Dense(500, activation='relu')(visible)
# hidden2 = Dense(100, activation='relu')(hidden1)
# hidden3 = Dense(50, activation='relu')(hidden2)
# hidden4 = Dense(20, activation='relu')(hidden3)
# output = Dense(3, activation='softmax')(hidden4)

# clf = Model(inputs=visible, outputs=output)
# print(clf.summary())

# from keras import metrics
# from keras import losses
# from keras import optimizers

# clf.compile(optimizer=optimizers.Adam(),
#               loss=losses.CategoricalCrossentropy(),
#               metrics=[metrics.Precision(), metrics.Recall()])

# train_predict_nn(clf, features, outcomes)