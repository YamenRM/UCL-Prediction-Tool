# src/tune_rf.py
import time
import pickle
import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
from train_test_split import load_split, compute_sample_weights
from Configuration import RF_TUNING_RESULTS_PATH, RF_MODEL_PATH
import os

os.makedirs(os.path.dirname(RF_TUNING_RESULTS_PATH), exist_ok=True)
os.makedirs(os.path.dirname(RF_MODEL_PATH), exist_ok=True)

X_train, y_train, X_val, y_val, train_df, val_df = load_split()
weights = compute_sample_weights(train_df['date'], reference_date=train_df['date'].max())

param_grid = {
    'n_estimators': [200, 300, 500, 800],
    'max_depth': [4, 5, 6, 8, 10, None],
    'min_samples_leaf': [5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.5],
}

# manuall RandomizedSearchCV so we can control our sample weights.
np.random.seed(42)
keys = list(param_grid.keys())
all_combos = list(product(*param_grid.values()))
n_iter = 40
sampled_combos = [all_combos[i] for i in np.random.choice(len(all_combos), size=n_iter, replace=False)]

results = []
best_score = np.inf
best_params = None
best_model = None

print(f"Searching {n_iter} random combinations out of {len(all_combos)} total...\n")

for i, combo in enumerate(sampled_combos, 1):
    params = dict(zip(keys, combo))
    start = time.time()

    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=weights)

    probs = model.predict_proba(X_val)
    preds = model.predict(X_val)
    loss = log_loss(y_val, probs, labels=model.classes_)
    acc = accuracy_score(y_val, preds)
    elapsed = time.time() - start

    results.append({**params, 'log_loss': round(loss, 4), 'accuracy': round(acc, 4), 'time_sec': round(elapsed, 2)})
    print(f"[{i}/{n_iter}] log_loss={loss:.4f} acc={acc:.4f} ({elapsed:.1f}s)  {params}")

    if loss < best_score:
        best_score = loss
        best_params = params
        best_model = model

print("\n" + "=" * 70)
print(f"BEST: log_loss={best_score:.4f}")
print(f"Params: {best_params}")

results_df = pd.DataFrame(results).sort_values('log_loss')
results_df.to_csv(RF_TUNING_RESULTS_PATH, index=False)
print(f"\n-> Saved full search results: {RF_TUNING_RESULTS_PATH}")


# Save the best model so train.py can just load it rather than retrain from scratch
with open(RF_MODEL_PATH, "wb") as f:
    pickle.dump({'model': best_model, 'params': best_params, 'log_loss': best_score}, f)
print(f"-> Saved best model: {RF_MODEL_PATH}")