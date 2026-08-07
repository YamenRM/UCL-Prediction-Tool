import pandas as pd
from rapidfuzz import process, fuzz
from Configuration import (
    PROCESSED_COMBINED_PATH, FIXTURES_TO_PREDICT_PATH,
    RAW_ELO_PATH, RAW_SOFIFA_PATH, Review_PATH
)

# Load our team names and reference team names from ELO and SoFIFA, then find the closest matches for each of our teams in both reference lists. The results are saved to a CSV for review.

played = pd.read_csv(PROCESSED_COMBINED_PATH)
unplayed = pd.read_csv(FIXTURES_TO_PREDICT_PATH)
our_teams = sorted(
    set(played['home_team']) | set(played['away_team'])
    | set(unplayed['home_team']) | set(unplayed['away_team'])
)


elo_teams = set(pd.read_csv(RAW_ELO_PATH)['team'])
sofifa_teams = set(pd.read_csv(RAW_SOFIFA_PATH)['team'])

candidates = []
for team in our_teams:
    elo_match, elo_score, _ = process.extractOne(team, elo_teams, scorer=fuzz.partial_ratio)
    sofifa_match, sofifa_score, _ = process.extractOne(team, sofifa_teams, scorer=fuzz.partial_ratio)
    candidates.append({
        'our_name': team,
        'elo_guess': elo_match, 'elo_score': elo_score,
        'sofifa_guess': sofifa_match, 'sofifa_score': sofifa_score,
    })

review_df = pd.DataFrame(candidates).sort_values(['sofifa_score', 'elo_score'])
review_df.to_csv(Review_PATH, index=False)
print(f"{len(review_df)} teams to review")
print(review_df.to_string())