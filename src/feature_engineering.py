import os
import numpy as np
import pandas as pd
from Configuration import (
    PROCESSED_WITH_TRANSFER_PATH,
    FIXTURES_WITH_TRANSFER_PATH,
    TRAINING_DATA_PATH,
    TO_PREDICT_DATA_PATH,
)

VALID_SEASONS = ['2122', '2223', '2324', '2425', '2526', '2627']


# keep only the seasons we actually asked for
def filter_valid_seasons(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df['season'].astype(str).isin(VALID_SEASONS)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"⚠️ Dropped {dropped} rows outside {VALID_SEASONS} (e.g. leaked historical seasons)")
    return df


# fill missing Elo / squad rating / squad delta values
def fill_missing_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Elo and squad rating -> fill with league/season average
    for col in ['home_elo', 'away_elo', 'home_squad_rating', 'away_squad_rating']:
        df[f'{col}_missing'] = df[col].isna()
        league_avg = df.groupby(['league', 'season'])[col].transform('mean')
        df[col] = df[col].fillna(league_avg)
        df[col] = df[col].fillna(df[col].mean())  # safety net for empty groups

    # Squad delta -> fill with league/season average, then 0 ("no known change")
    for col in ['home_squad_delta', 'away_squad_delta']:
        df[f'{col}_missing'] = df[col].isna()
        league_avg = df.groupby(['league', 'season'])[col].transform('mean')
        df[col] = df[col].fillna(league_avg)
        df[col] = df[col].fillna(0)

    df['elo_diff'] = df['home_elo'] - df['away_elo']
    df['squad_rating_diff'] = df['home_squad_rating'] - df['away_squad_rating']
    df['squad_delta_diff'] = df['home_squad_delta'] - df['away_squad_delta']
    return df


# Rolling form (points + goal difference), leakage-safe
def add_rolling_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values('date').copy()

    home = df[['date', 'home_team', 'home_score', 'away_score']].rename(
        columns={'home_team': 'team', 'home_score': 'goals_for', 'away_score': 'goals_against'})
    away = df[['date', 'away_team', 'away_score', 'home_score']].rename(
        columns={'away_team': 'team', 'away_score': 'goals_for', 'home_score': 'goals_against'})

    for part in (home, away):
        part['points'] = np.select(
            [part['goals_for'] > part['goals_against'], part['goals_for'] == part['goals_against']],
            [3, 1], default=0
        )

    long_form = pd.concat([home, away]).sort_values(['team', 'date'])

    def rolling_shifted(series):
        return series.shift(1).rolling(window, min_periods=1).mean()

    long_form['rolling_points'] = long_form.groupby('team')['points'].transform(rolling_shifted)
    long_form['rolling_gf'] = long_form.groupby('team')['goals_for'].transform(rolling_shifted)
    long_form['rolling_ga'] = long_form.groupby('team')['goals_against'].transform(rolling_shifted)
    long_form['rolling_gd'] = long_form['rolling_gf'] - long_form['rolling_ga']

    form_lookup = long_form[['team', 'date', 'rolling_points', 'rolling_gd']].sort_values('date')

    df = pd.merge_asof(
        df.sort_values('date'),
        form_lookup.rename(columns={'team': 'home_team', 'rolling_points': 'home_form_pts', 'rolling_gd': 'home_form_gd'}),
        on='date', by='home_team', direction='backward'
    )
    df = pd.merge_asof(
        df.sort_values('date'),
        form_lookup.rename(columns={'team': 'away_team', 'rolling_points': 'away_form_pts', 'rolling_gd': 'away_form_gd'}),
        on='date', by='away_team', direction='backward'
    )

    for col in ['home_form_pts', 'away_form_pts']:
        df[col] = df[col].fillna(1.5)
    for col in ['home_form_gd', 'away_form_gd']:
        df[col] = df[col].fillna(0)

    df['form_pts_diff'] = df['home_form_pts'] - df['away_form_pts']
    df['form_gd_diff'] = df['home_form_gd'] - df['away_form_gd']
    return df



# rest days since each team's last match
def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('date').copy()

    for side in ['home_team', 'away_team']:
        long_dates = pd.concat([
            df[['date', 'home_team']].rename(columns={'home_team': 'team'}),
            df[['date', 'away_team']].rename(columns={'away_team': 'team'})
        ]).sort_values(['team', 'date'])

        long_dates['prev_date'] = long_dates.groupby('team')['date'].shift(1)
        long_dates['rest_days'] = (long_dates['date'] - long_dates['prev_date']).dt.days
        long_dates = long_dates.sort_values('date')

        col_name = f'{side}_rest_days'
        merged = pd.merge_asof(
            df.sort_values('date'),
            long_dates[['team', 'date', 'rest_days']].rename(columns={'team': side, 'rest_days': col_name}),
            on='date', by=side, direction='backward'
        )
        df[col_name] = merged[col_name].fillna(7)

    df['rest_days_diff'] = df['home_team_rest_days'] - df['away_team_rest_days']
    return df



# target label (only meaningful for played matches) 
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    conditions = [df['home_score'] > df['away_score'], df['home_score'] == df['away_score']]
    df['result'] = np.select(conditions, ['H', 'D'], default='A')
    return df



# Pipeline
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = fill_missing_ratings(df)
    df = add_rolling_form(df)
    df = add_rest_days(df)
    return df

# Main execution
if __name__ == "__main__":
    played = pd.read_csv(PROCESSED_WITH_TRANSFER_PATH)
    unplayed = pd.read_csv(FIXTURES_WITH_TRANSFER_PATH)

    played = filter_valid_seasons(played)
    unplayed = filter_valid_seasons(unplayed)

    played['_source'] = 'played'
    unplayed['_source'] = 'unplayed'

    print(f"Building features for {len(played)} played matches...")
    combined = pd.concat([played, unplayed], ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    combined = build_features(combined)

    played_final = combined[combined['_source'] == 'played'].drop(columns=['_source']).copy()
    played_final = add_target(played_final)
    unplayed_final = combined[combined['_source'] == 'unplayed'].drop(columns=['_source']).copy()

    assert len(played_final) == len(played), f"Row count mismatch: {len(played_final)} vs {len(played)}"
    assert len(unplayed_final) == len(unplayed), f"Row count mismatch: {len(unplayed_final)} vs {len(unplayed)}"

    os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TO_PREDICT_DATA_PATH), exist_ok=True)

    played_final.to_csv(TRAINING_DATA_PATH, index=False)
    unplayed_final.to_csv(TO_PREDICT_DATA_PATH, index=False)

    print(f"\n-> Saved training data: {TRAINING_DATA_PATH} ({len(played_final)} rows)")
    print(f"-> Saved to-predict data: {TO_PREDICT_DATA_PATH} ({len(unplayed_final)} rows)")
    print(f"\nColumns: {played_final.columns.tolist()}")
    print(f"\nResult distribution:\n{played_final['result'].value_counts(normalize=True)}")
    print(f"\nRemaining NaN check:\n{played_final.isna().sum()[played_final.isna().sum() > 0]}")