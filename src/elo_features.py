import time
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
    elo_reader = sd.ClubElo()
    elo_reader.rate_limit = 3
    elo_reader.max_delay = 2

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
            if histories:
                pd.concat(histories, ignore_index=True).to_csv(RAW_ELO_PATH + ".partial", index=False)

    if failed:
        print(f"⚠️ Could not fetch Elo history for {len(failed)} teams:")
        print(failed)

    if not histories:  # <-- guard
        return pd.DataFrame(columns=['team', 'from', 'elo'])

    elo_long = pd.concat(histories, ignore_index=True)
    elo_long['from'] = pd.to_datetime(elo_long['from'])
    elo_long = elo_long[['team', 'from', 'elo']].sort_values(['team', 'from'])
    return elo_long


def add_elo_features(matches: pd.DataFrame, elo_long: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches['date'] = pd.to_datetime(matches['date'])
    matches['home_team'] = matches['home_team'].astype(str)
    matches['away_team'] = matches['away_team'].astype(str)
    matches = matches.sort_values('date')

    elo_long = elo_long.copy()
    elo_long['team'] = elo_long['team'].astype(str)
    elo_long = elo_long.sort_values('from')

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

    try:
        existing = pd.read_csv(RAW_ELO_PATH, parse_dates=['from'])   
        already_have = set(existing['team'].unique())
    except FileNotFoundError:
        existing = pd.DataFrame(columns=['team', 'from', 'elo'])
        already_have = set()

    teams_to_fetch = [t for t in all_teams if t not in already_have]
    print(f"{len(already_have)} teams already cached, fetching {len(teams_to_fetch)} new/missing teams...")

    new_elo = fetch_elo_history(teams_to_fetch) if teams_to_fetch else pd.DataFrame(columns=['team', 'from', 'elo'])

    elo_long = pd.concat([existing, new_elo], ignore_index=True)
    elo_long['from'] = pd.to_datetime(elo_long['from'], format='mixed')  
    elo_long = elo_long.drop_duplicates(subset=['team', 'from'])
    elo_long.to_csv(RAW_ELO_PATH, index=False)

    played_elo = add_elo_features(played, elo_long)
    unplayed_elo = add_elo_features(unplayed, elo_long)

    played_elo.to_csv(PROCESSED_WITH_ELO_PATH, index=False)
    unplayed_elo.to_csv(FIXTURES_WITH_ELO_PATH, index=False)

    missing_home = played_elo['home_elo'].isna().sum()
    missing_away = played_elo['away_elo'].isna().sum()
    print(f"\nMissing home_elo: {missing_home} / {len(played_elo)}")
    print(f"Missing away_elo: {missing_away} / {len(played_elo)}")
    print(f"\n-> Saved: {PROCESSED_WITH_ELO_PATH}")
    print(f"-> Saved: {FIXTURES_WITH_ELO_PATH}")