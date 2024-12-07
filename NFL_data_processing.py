import pandas as pd
import numpy as np

df = pd.read_csv('nfl.csv')

df.columns = [
    'Rk', 'Team', 'Date', 'Season', 'Pts', 'PtsO', 'Team_Rate', 'Team_TO', 'Y/P', 'DY/P',
    'Team_ToP', 'Opp_Rate', 'Day', 'G#', 'Week', 'Away', 'Opp', 'Result', 'Pts_dup', 'PtsO_dup',
    'PtDif', 'PC', 'Cmp', 'Att', 'Inc', 'Cmp%', 'Yds', 'TD', 'Int', 'TD%', 'Int%_dup',
    'Opp_Rate_dup', 'Sk', 'Yds_Sk', 'Sk%', 'Y/A', 'NY/A', 'AY/A', 'ANY/A', 'Y/C', 'Tot',
    'Ply', 'Y/P_dup', 'DPly', 'DY/P_dup', 'TO_dup', 'ToP_dup', 'Time', 'Opp_Cmp', 'Opp_Att',
    'Opp_Cmp%', 'Opp_Yds', 'Opp_TD', 'Opp_Sk', 'Opp_Yds_Sk', 'Opp_Int', 'Opp_Rate_opp',
    'Rush', 'Pass', 'Tot_Yds', 'TO_game'
]

df = df.loc[:, ~df.columns.duplicated()]

df['Home'] = df['Away'].apply(lambda x: 0 if x == '@' else 1)
df.drop('Away', axis=1, inplace=True)

def parse_outcome(result):
    if pd.isnull(result):
        return np.nan
    result = result.strip()
    if result.startswith('W'):
        return 1  # Win
    elif result.startswith('L'):
        return 0  # Loss
    else:
        return np.nan

df['Outcome'] = df['Result'].apply(parse_outcome)
df.drop('Result', axis=1, inplace=True)

def convert_time(time_str):
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes + seconds / 60.0
    except:
        return np.nan

df['Team_ToP'] = df['Team_ToP'].apply(convert_time)

numeric_cols = ['Team_Rate', 'Opp_Rate', 'Team_TO', 'Y/P', 'DY/P', 'Pts', 'PtsO', 'Tot', 'Ply', 'Rush', 'Pass', 'Tot_Yds']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

df['Net_YPP'] = df['Y/P'] - df['DY/P']  # Net Yards Per Play
df['Net_Pts'] = df['Pts'] - df['PtsO']  # Net Points
df['Pts_per_Ply'] = df['Pts'] / df['Ply']  # Points Per Play
df['TO_Margin'] = df['Team_TO'] - df['TO_game']  # Turnover Margin
df['Pass_YPA'] = df['Pass'] / (df['Att'] + 1e-5)  # Pass Yards Per Attempt
df['Rush_YPA'] = df['Rush'] / (df['Ply'] - df['Att'] + 1e-5)  # Rush Yards Per Attempt
df['ToP_Diff'] = df['Team_ToP'] - df['ToP_dup'].apply(convert_time)  # Time of Possession Differential


selected_features = [
    'Team',         # Team name
    'Home',         # Home field indicator
    'Team_Rate',    # Team QB passer rating
    'Opp_Rate',     # Opponent QB passer rating
    'Pts_per_Ply',  # Points Per Play
    'TO_Margin',    # Turnover Margin
    'Net_YPP',      # Net yards per play
    'Pts',
    'PtsO',
    'Pass_YPA',     # Pass Yards Per Attempt
    'Rush_YPA',     # Rush Yards Per Attempt
    'ToP_Diff'      # Time of Possession Differential
]

final_df = df[selected_features + ['Outcome']]

final_df.dropna(inplace=True)

final_df.reset_index(drop=True, inplace=True)

final_df = pd.get_dummies(final_df, columns=['Team'], prefix='Team')

print(final_df.head())

final_df.to_csv('nfl_games_cleaned.csv', index=False)
