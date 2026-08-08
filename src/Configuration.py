import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DOMESTIC_PATH = f"{PROJECT_PATH}/data/raw/matches_domestic.csv"
RAW_UCL_PATH = f"{PROJECT_PATH}/data/raw/matches_ucl.csv"
PROCESSED_COMBINED_PATH = f"{PROJECT_PATH}/data/processed/matches_combined.csv"
FIXTURES_TO_PREDICT_PATH = f"{PROJECT_PATH}/data/processed/fixtures_to_predict.csv"

RAW_ELO_PATH = f"{PROJECT_PATH}/data/external/team_elo_history.csv"
PROCESSED_WITH_ELO_PATH = f"{PROJECT_PATH}/data/processed/matches_with_elo.csv"
FIXTURES_WITH_ELO_PATH = f"{PROJECT_PATH}/data/processed/fixtures_with_elo.csv"

RAW_SOFIFA_PATH = f"{PROJECT_PATH}/data/external/sofifa_team_ratings.csv"
PROCESSED_WITH_TRANSFER_PATH = f"{PROJECT_PATH}/data/processed/matches_with_transfer.csv"
FIXTURES_WITH_TRANSFER_PATH = f"{PROJECT_PATH}/data/processed/fixtures_with_transfer.csv"

Review_PATH = f"{PROJECT_PATH}/data/external/team_name_review.csv"
json_mapping_path = f"{PROJECT_PATH}/data/external/teamname_replacements_auto.json"
Need_Review_PATH = f"{PROJECT_PATH}/data/external/team_name_needs_review.csv"

TRAINING_DATA_PATH = f"{PROJECT_PATH}/data/final/training_data.csv"
TO_PREDICT_DATA_PATH = f"{PROJECT_PATH}/data/final/to_predict_data.csv"