"""
Training two Random Forest models:

1. Evaluation model (2122-2425) — deliberately holds out the 2526 season
   so evaluate.py can report honest, unseen-data performance metrics.

2. Production model (2122-2526) — trained on all available real match
   history, used by predict.py to score the live 2627 fixtures.

"""

import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from Configuration import TRAINING_DATA_PATH, MODELS_DIR
from train_test_split import FEATURE_COLS, TARGET_COL, compute_sample_weights

EVAL_MODEL_PATH = f"{MODELS_DIR}/rf_eval.pkl"
PRODUCTION_MODEL_PATH = f"{MODELS_DIR}/rf_final.pkl"

os.makedirs(MODELS_DIR, exist_ok=True)

BEST_PARAMS = {
    'n_estimators': 800,
    'max_depth': 6,
    'min_samples_leaf': 20,
    'min_samples_split': 10,
    'max_features': 'log2',
}

EVAL_SEASONS = ['2122', '2223', '2324', '2425']
PRODUCTION_SEASONS = ['2122', '2223', '2324', '2425', '2526']


def load_seasons(seasons: list[str]) -> pd.DataFrame:
    df = pd.read_csv(TRAINING_DATA_PATH, parse_dates=['date'])
    df['season'] = df['season'].astype(str)
    return df[df['season'].isin(seasons)].copy()


def train_and_save(df: pd.DataFrame, save_path: str, label: str) -> RandomForestClassifier:
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    weights = compute_sample_weights(df['date'], reference_date=df['date'].max())

    model = RandomForestClassifier(**BEST_PARAMS, random_state=42, n_jobs=-1)
    model.fit(X, y, sample_weight=weights)

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    with open(save_path, "wb") as f:
        pickle.dump({
            'model': model,
            'feature_cols': FEATURE_COLS,
            'params': BEST_PARAMS,
            'trained_on_seasons': sorted(df['season'].unique()),
            'trained_on_rows': len(df),
        }, f)

    print(f"\n{label}")
    print(f"  Trained on {len(df)} rows, seasons: {sorted(df['season'].unique())}")
    print(f"  Classes: {model.classes_}")
    print(f"  Top 5 features:\n{importances.head(5).to_string()}")
    print(f"  -> Saved: {save_path}")

    return model


if __name__ == "__main__":

    os.makedirs(os.path.dirname(EVAL_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PRODUCTION_MODEL_PATH), exist_ok=True)

    eval_df = load_seasons(EVAL_SEASONS)
    train_and_save(eval_df, EVAL_MODEL_PATH, "Evaluation model (holds out 2526)")

    production_df = load_seasons(PRODUCTION_SEASONS)
    train_and_save(production_df, PRODUCTION_MODEL_PATH, "Production model (all history through 2526)")

    print(f"\nDone. Use rf_eval.pkl in evaluate.py, rf_final.pkl in predict.py.")