import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DOMESTIC_PATH = f"{PROJECT_PATH}/data/raw/matches_domestic.csv"
RAW_UCL_PATH = f"{PROJECT_PATH}/data/raw/matches_ucl.csv"
PROCESSED_COMBINED_PATH = f"{PROJECT_PATH}/data/processed/matches_combined.csv"
FIXTURES_TO_PREDICT_PATH = f"{PROJECT_PATH}/data/processed/fixtures_to_predict.csv"
