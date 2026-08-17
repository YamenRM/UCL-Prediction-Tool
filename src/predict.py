import pickle
import numpy as np
import pandas as pd

from Configuration import TO_PREDICT_DATA_PATH, PRODUCTION_MODEL_PATH, PREDICTIONS_OUTPUT_PATH


DRAW_MARGIN = 0.08


def load_model(path: str):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle['model'], bundle['feature_cols']


def load_fixtures() -> pd.DataFrame:
    df = pd.read_csv(TO_PREDICT_DATA_PATH, parse_dates=['date'])
    df['season'] = df['season'].astype(str)
    return df

# Draw margin rule: if the probability of a Draw is within DRAW_MARGIN of the top predicted class, call it a Draw instead. This is used to avoid overconfident predictions in close matches.
def predict_with_draw_margin(probs: np.ndarray, classes: np.ndarray, margin: float = DRAW_MARGIN) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]  
    top_prob = sorted_probs[:, 0]
    second_prob = sorted_probs[:, 1]

    top_idx = probs.argmax(axis=1)
    preds = classes[top_idx]

    too_close_to_call = (top_prob - second_prob) <= margin
    preds = np.where(too_close_to_call, 'D', preds)
    return preds


if __name__ == "__main__":
    model, feature_cols = load_model(PRODUCTION_MODEL_PATH)
    print(f"Loaded production model ({len(feature_cols)} features)")

    fixtures = load_fixtures()
    print(f"Scoring {len(fixtures)} unplayed fixtures...")

    missing_cols = [c for c in feature_cols if c not in fixtures.columns]
    if missing_cols:
        raise ValueError(f"Fixtures data is missing expected feature columns: {missing_cols}")

    X = fixtures[feature_cols]
    probs = model.predict_proba(X)
    classes = model.classes_

    predicted_label = predict_with_draw_margin(probs, classes)
    predicted_label_argmax = model.predict(X) 

    output = fixtures[['date', 'league', 'season', 'home_team', 'away_team',
                    'home_elo_missing', 'away_elo_missing',
                    'home_squad_rating_missing', 'away_squad_rating_missing']].copy()
    for i, cls in enumerate(classes):
        output[f'prob_{cls}'] = probs[:, i].round(4)

    output['predicted'] = predicted_label
    output['predicted_argmax'] = predicted_label_argmax
    output['confidence'] = probs.max(axis=1).round(4)

    output = output.sort_values('date')
    output.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)

    print(f"\n-> Saved predictions: {PREDICTIONS_OUTPUT_PATH}")
    print(f"\nPredicted outcome distribution (draw-margin rule):")
    print(output['predicted'].value_counts(normalize=True).round(3))
    print(f"\nPredicted outcome distribution (plain argmax, for comparison):")
    print(output['predicted_argmax'].value_counts(normalize=True).round(3))
    BIG5_LEAGUES = ['ENG-Premier League', 'ESP-La Liga', 'FRA-Ligue 1', 'GER-Bundesliga', 'ITA-Serie A']

    big5_output = output[output['league'].isin(BIG5_LEAGUES)]

    print(f"\nNext 10 Big-5 league fixtures:")
    print(big5_output[['date', 'league', 'home_team', 'away_team',
                    'prob_H', 'prob_D', 'prob_A', 'predicted']].head(10).to_string(index=False))