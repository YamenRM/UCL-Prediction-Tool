import pandas as pd
import soccerdata as sd
from Configuration import (
    PROCESSED_WITH_ELO_PATH,
    FIXTURES_WITH_ELO_PATH,
    RAW_SOFIFA_PATH,
    PROCESSED_WITH_TRANSFER_PATH,
    FIXTURES_WITH_TRANSFER_PATH,
)

# Fetch SoFIFA team ratings 
def fetch_sofifa_ratings(leagues=['ENG-Premier League', 'ESP-La Liga', 'FRA-Ligue 1', 'GER-Bundesliga', 'ITA-Serie A']) -> pd.DataFrame:

    sofifa = sd.SoFIFA(leagues=leagues, versions=[210061, 220030, 230001, 230013, 240002, 240024, 250002, 250019, 260002, 260020, 260046])
    ratings = sofifa.read_team_ratings().reset_index()
    ratings['update'] = pd.to_datetime(ratings['update'], errors='coerce')
    ratings['overall'] = pd.to_numeric(ratings['overall'], errors='coerce')
    return ratings.sort_values(['team', 'update'])

# Add transfer features to matches DataFrame
def add_transfer_features(matches: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:

    ratings = ratings.copy()
    ratings['prev_overall'] = ratings.groupby('team')['overall'].shift(1)
    ratings['delta'] = ratings['overall'] - ratings['prev_overall']
    ratings = ratings[['team', 'update', 'overall', 'delta']].rename(columns={'update': 'date'})
    ratings = ratings.sort_values('date')  

    matches = matches.copy()
    matches['date'] = pd.to_datetime(matches['date'])
    matches = matches.sort_values('date')

    home = pd.merge_asof(
        matches,
        ratings.rename(columns={'team': 'home_team', 'overall': 'home_squad_rating',
                                 'delta': 'home_squad_delta'}),
        on='date', by='home_team', direction='backward'
    )
    merged = pd.merge_asof(
        home.sort_values('date'),
        ratings.rename(columns={'team': 'away_team', 'overall': 'away_squad_rating',
                                 'delta': 'away_squad_delta'}),
        on='date', by='away_team', direction='backward'
    )
    return merged


if __name__ == "__main__":
    print("Fetching SoFIFA team ratings")
    ratings = fetch_sofifa_ratings()
    ratings.to_csv(RAW_SOFIFA_PATH, index=False)
    print(f"-> Saved raw ratings: {RAW_SOFIFA_PATH}")

    played = pd.read_csv(PROCESSED_WITH_ELO_PATH)
    unplayed = pd.read_csv(FIXTURES_WITH_ELO_PATH)

    played_full = add_transfer_features(played, ratings)
    unplayed_full = add_transfer_features(unplayed, ratings)

    played_full.to_csv(PROCESSED_WITH_TRANSFER_PATH, index=False)
    unplayed_full.to_csv(FIXTURES_WITH_TRANSFER_PATH, index=False)

    missing_home = played_full['home_squad_rating'].isna().sum()
    missing_away = played_full['away_squad_rating'].isna().sum()
    print(f"\nMissing home_squad_rating: {missing_home} / {len(played_full)}")
    print(f"Missing away_squad_rating: {missing_away} / {len(played_full)}")
    print(f"\n-> Saved: {PROCESSED_WITH_TRANSFER_PATH}")
    print(f"-> Saved: {FIXTURES_WITH_TRANSFER_PATH}")