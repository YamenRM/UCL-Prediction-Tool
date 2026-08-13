import pandas as pd
from Configuration import TRAINING_DATA_PATH

FEATURE_COLS = [
    'elo_diff', 'home_elo', 'away_elo',
    'squad_rating_diff', 'home_squad_rating', 'away_squad_rating',
    'squad_delta_diff', 'home_squad_delta', 'away_squad_delta',
    'form_pts_diff', 'home_form_pts', 'away_form_pts',
    'form_gd_diff', 'home_form_gd', 'away_form_gd',
    'rest_days_diff', 'home_team_rest_days', 'away_team_rest_days',
    'home_elo_missing', 'away_elo_missing',
    'home_squad_rating_missing', 'away_squad_rating_missing',
    'home_squad_delta_missing', 'away_squad_delta_missing',
]
TARGET_COL = 'result'

TRAIN_SEASONS = ['2122', '2223', '2324', '2425']
VAL_SEASON = '2526'

def load_split():
    df = pd.read_csv(TRAINING_DATA_PATH, parse_dates=['date'])
    df['season'] = df['season'].astype(str)

    train_df = df[df['season'].isin(TRAIN_SEASONS)].copy()
    val_df = df[df['season'] == VAL_SEASON].copy()

    print(f"Train: {len(train_df)} rows ({TRAIN_SEASONS})")
    print(f"Val:   {len(val_df)} rows ({VAL_SEASON})")

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET_COL]

    return X_train, y_train, X_val, y_val, train_df, val_df

# Compute sample weights based on recency
def compute_sample_weights(dates: pd.Series, reference_date, half_life_days: int = 500) -> pd.Series:
    days_old = (reference_date - dates).dt.days
    return 0.5 ** (days_old / half_life_days)


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, train_df, val_df = load_split()

    weights = compute_sample_weights(train_df['date'], reference_date=train_df['date'].max())
    print(f"\nSample weight range: {weights.min():.4f} to {weights.max():.4f}")
    print(f"Oldest match weight: {weights.min():.4f} (from {train_df.loc[weights.idxmin(), 'date']})")
    print(f"Newest match weight: {weights.max():.4f} (from {train_df.loc[weights.idxmax(), 'date']})")

    print(f"\nTrain result distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"\nVal result distribution:\n{y_val.value_counts(normalize=True)}")