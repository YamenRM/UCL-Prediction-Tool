import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss, accuracy_score, confusion_matrix,
    classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve

from Configuration import TRAINING_DATA_PATH, MODELS_DIR
from train_test_split import FEATURE_COLS, TARGET_COL

EVAL_MODEL_PATH = f"{MODELS_DIR}/rf_eval.pkl"
EVAL_SEASON = '2526'


def load_eval_data():
    df = pd.read_csv(TRAINING_DATA_PATH, parse_dates=['date'])
    df['season'] = df['season'].astype(str)
    return df[df['season'] == EVAL_SEASON].copy()


if __name__ == "__main__":
    with open(EVAL_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle['model']
    print(f"Loaded model trained on: {bundle['trained_on_seasons']} ({bundle['trained_on_rows']} rows)")

    eval_df = load_eval_data()
    X_eval = eval_df[FEATURE_COLS]
    y_eval = eval_df[TARGET_COL]
    print(f"Evaluating on {len(eval_df)} genuinely unseen matches from season {EVAL_SEASON}\n")

    probs = model.predict_proba(X_eval)
    preds = model.predict(X_eval)
    prob_true, prob_pred = calibration_curve((y_eval == 'D').astype(int), probs[:, 1], n_bins=10)


    # Core metrics 
    acc = accuracy_score(y_eval, preds)
    loss = log_loss(y_eval, probs, labels=model.classes_)

    naive_acc = accuracy_score(y_eval, ['H'] * len(y_eval))

    print("=" * 60)
    print(f"Accuracy:        {acc:.4f}  (naive 'always Home': {naive_acc:.4f})")
    print(f"Log loss:        {loss:.4f}")
    print("=" * 60)

    # Confusion matrix
    cm = confusion_matrix(y_eval, preds, labels=model.classes_)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in model.classes_],
                          columns=[f"pred_{c}" for c in model.classes_])
    print("\nConfusion matrix:")
    print(cm_df)

    # classification report
    print("\nClassification report:")
    print(classification_report(y_eval, preds, labels=model.classes_))

    # Brier score per class (calibration) 
    print("Brier scores (calibration, lower is better):")
    for i, cls in enumerate(model.classes_):
        y_true_binary = (y_eval == cls).astype(int)
        brier = brier_score_loss(y_true_binary, probs[:, i])
        print(f"  {cls}: {brier:.4f}")

    # Save predictions alongside actuals for further inspection 
    output = eval_df[['date', 'league', 'home_team', 'away_team', 'result']].copy()
    for i, cls in enumerate(model.classes_):
        output[f'prob_{cls}'] = probs[:, i]
    output['predicted'] = preds
    output['correct'] = output['predicted'] == output['result']

    output_path = f"{MODELS_DIR}/eval_predictions_2526.csv"
    output.to_csv(output_path, index=False)
    print(f"\n-> Saved per-match predictions: {output_path}")

    # Worst misses — biggest confidence on a wrong prediction 
    output['confidence'] = output[[f'prob_{c}' for c in model.classes_]].max(axis=1)
    worst_misses = output[~output['correct']].sort_values('confidence', ascending=False).head(10)
    print("\nMost confident wrong predictions:")
    print(worst_misses[['date', 'home_team', 'away_team', 'result', 'predicted', 'confidence']].to_string(index=False))

    # Calibration check for the draw class
    draw_idx = list(model.classes_).index('D')
    y_is_draw = (y_eval == 'D').astype(int)
    draw_probs = probs[:, draw_idx]

    prob_true, prob_pred = calibration_curve(y_is_draw, draw_probs, n_bins=10, strategy='quantile')

    print("\nCalibration check — Draw class:")
    print("(predicted probability bin  vs  actual draw frequency in that bin)")
    for pt, pp in zip(prob_true, prob_pred):
        print(f"  predicted ~{pp:.3f}  ->  actual {pt:.3f}")

    # Save the raw calibration data too, useful for plotting later
    calib_df = pd.DataFrame({'predicted_prob': prob_pred, 'actual_freq': prob_true})
    calib_path = f"{MODELS_DIR}/draw_calibration.csv"
    calib_df.to_csv(calib_path, index=False)
    print(f"\n-> Saved calibration data: {calib_path}")

