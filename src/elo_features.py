import pandas as pd
import soccerdata as sd
from Configuration import (
    PROCESSED_COMBINED_PATH,
    FIXTURES_TO_PREDICT_PATH,
    RAW_ELO_PATH,
    PROCESSED_WITH_ELO_PATH,
    FIXTURES_WITH_ELO_PATH,
)

def fetch_elo_history(teams: list[str]) -> pd.DataFrame:
    # Pull full Elo rating history for each team from clubelo.com.
    elo_reader = sd.ClubElo()
    histories = []
    failed = []

    for i, team in enumerate(teams, 1):
        try:
            hist = elo_reader.read_team_history(team).reset_index()
            hist['team'] = team 
            histories.append(hist)
        except Exception:
            failed.append(team)
        if i % 20 == 0:
            print(f"  [{i}/{len(teams)}] fetched")

    if failed:
        print(f"⚠️ Could not fetch Elo history for {len(failed)} teams:")
        print(failed)

    # Concatenate all histories into a single DataFrame
    elo_long = pd.concat(histories, ignore_index=True)
    elo_long['from'] = pd.to_datetime(elo_long['from'])
    elo_long = elo_long[['team', 'from', 'elo']].sort_values(['team', 'from'])
    return elo_long


def add_elo_features(matches: pd.DataFrame, elo_long: pd.DataFrame) -> pd.DataFrame:
    # Add Elo features to the matches DataFrame by merging with the Elo history.
    matches = matches.copy()
    matches['date'] = pd.to_datetime(matches['date'])
    matches = matches.sort_values('date')

    home_elo = pd.merge_asof(
        matches,
        elo_long.rename(columns={'team': 'home_team', 'from': 'date', 'elo': 'home_elo'}),
        on='date', by='home_team', direction='backward'
    )
    merged = pd.merge_asof(
        home_elo.sort_values('date'),
        elo_long.rename(columns={'team': 'away_team', 'from': 'date', 'elo': 'away_elo'}),
        on='date', by='away_team', direction='backward'
    )
    merged['elo_diff'] = merged['home_elo'] - merged['away_elo']
    return merged


if __name__ == "__main__":
    played = pd.read_csv(PROCESSED_COMBINED_PATH)
    unplayed = pd.read_csv(FIXTURES_TO_PREDICT_PATH)

    all_teams = sorted(
        set(played['home_team']) | set(played['away_team'])
        | set(unplayed['home_team']) | set(unplayed['away_team'])
    )
    print(f"Fetching Elo history for {len(all_teams)} teams...")
    elo_long = fetch_elo_history(all_teams)
    elo_long.to_csv(RAW_ELO_PATH, index=False)

    played_elo = add_elo_features(played, elo_long)
    unplayed_elo = add_elo_features(unplayed, elo_long)

    played_elo.to_csv(PROCESSED_WITH_ELO_PATH, index=False)
    unplayed_elo.to_csv(FIXTURES_WITH_ELO_PATH, index=False)

    # Report missing Elo values
    missing_home = played_elo['home_elo'].isna().sum()
    missing_away = played_elo['away_elo'].isna().sum()
    print(f"\nMissing home_elo: {missing_home} / {len(played_elo)}")
    print(f"Missing away_elo: {missing_away} / {len(played_elo)}")
    print(f"\n-> Saved: {PROCESSED_WITH_ELO_PATH}")
    print(f"-> Saved: {FIXTURES_WITH_ELO_PATH}")