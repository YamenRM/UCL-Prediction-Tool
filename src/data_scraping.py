import pandas as pd
import soccerdata as sd
from Configuration import RAW_DOMESTIC_PATH, RAW_UCL_PATH, PROCESSED_COMBINED_PATH, FIXTURES_TO_PREDICT_PATH

SEASONS = ['2122', '2223', '2324', '2425', '2526', '2627']


# Scrape & Process Domestic Data
print("Scraping Domestic Leagues from FBref...")
fbref = sd.FBref(leagues='Big 5 European Leagues Combined', seasons=SEASONS, headless=False)
fbref.rate_limit = 12
schedule_domestic = fbref.read_schedule().reset_index()  
schedule_domestic = schedule_domestic[schedule_domestic['season'].isin(SEASONS)]

# split "score" ("2–1", en dash) into home_score / away_score
if 'score' in schedule_domestic.columns:
    scores = schedule_domestic['score'].str.split('–', expand=True)
    schedule_domestic['home_score'] = pd.to_numeric(scores[0], errors='coerce')
    schedule_domestic['away_score'] = pd.to_numeric(scores[1], errors='coerce')

cols_domestic = ['league', 'season', 'game_id', 'date', 'home_team', 'away_team','home_score', 'away_score', 'home_xg', 'away_xg']
missing = [c for c in cols_domestic if c not in schedule_domestic.columns]
if missing:
    print(f"⚠️ Missing expected columns: {missing}")
schedule_domestic = schedule_domestic[[c for c in cols_domestic if c in schedule_domestic.columns]].copy()

played_domestic = schedule_domestic.dropna(subset=['home_score', 'away_score']).copy()
unplayed_domestic = schedule_domestic[
    schedule_domestic['home_score'].isna() & (schedule_domestic['season'] == '2627')
].copy()

played_domestic.to_csv(RAW_DOMESTIC_PATH, index=False)

# Scrape & Process UCL Data
print("Scraping Champions League from FBref...")
fbref_ucl = sd.FBref(leagues=['INT-Champions League'], seasons=SEASONS, headless=False)
fbref_ucl.rate_limit = 12
schedule_ucl = fbref_ucl.read_schedule().reset_index()

if 'score' in schedule_ucl.columns:
    scores = schedule_ucl['score'].str.split('–', expand=True)
    schedule_ucl['home_score'] = pd.to_numeric(scores[0], errors='coerce')
    schedule_ucl['away_score'] = pd.to_numeric(scores[1], errors='coerce')

schedule_ucl = schedule_ucl[[c for c in cols_domestic if c in schedule_ucl.columns]].copy()

played_ucl = schedule_ucl.dropna(subset=['home_score', 'away_score']).copy()
unplayed_ucl = schedule_ucl[schedule_ucl['home_score'].isna() & (schedule_ucl['season'] == '2627')].copy()
played_ucl.to_csv(RAW_UCL_PATH, index=False)

# Merge Domestic and UCL Data 
master_df = pd.concat([played_domestic, played_ucl], ignore_index=True)
master_df.to_csv(PROCESSED_COMBINED_PATH, index=False)

fixtures_to_predict = pd.concat([unplayed_domestic, unplayed_ucl], ignore_index=True)
fixtures_to_predict.to_csv(FIXTURES_TO_PREDICT_PATH, index=False)

print(f"\nPipeline Complete!")
print(f"-> Historical Training Data: {PROCESSED_COMBINED_PATH} ({len(master_df)} matches)")
print(f"-> Unplayed 2026/27 Fixtures: {FIXTURES_TO_PREDICT_PATH} ({len(fixtures_to_predict)} matches)")