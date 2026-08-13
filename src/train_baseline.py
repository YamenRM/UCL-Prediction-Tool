import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from train_test_split import load_split, compute_sample_weights

X_train, y_train, X_val, y_val, train_df, val_df = load_split()
weights = compute_sample_weights(train_df['date'], reference_date=train_df['date'].max())

# Naive baseline: always predict the majority class
majority_class = y_train.value_counts().idxmax()
naive_preds = [majority_class] * len(y_val)
naive_acc = accuracy_score(y_val, naive_preds)
print(f"Naive baseline (always predict '{majority_class}'): {naive_acc:.4f} accuracy")

# baseline: logistic regression using only elo_diff
X_train_simple = X_train[['elo_diff']]
X_val_simple = X_val[['elo_diff']]

model = LogisticRegression(max_iter=1000)
model.fit(X_train_simple, y_train, sample_weight=weights)

val_preds = model.predict(X_val_simple)
val_probs = model.predict_proba(X_val_simple)

acc = accuracy_score(y_val, val_preds)
loss = log_loss(y_val, val_probs, labels=model.classes_)

print(f"\nLogistic Regression (elo_diff only):")
print(f"  Accuracy: {acc:.4f}")
print(f"  Log loss: {loss:.4f}")
print(f"  Classes: {model.classes_}")
print(f"  Coefficient: {model.coef_}")

# Compute Brier scores for each class
for i, cls in enumerate(model.classes_):
    y_true_binary = (y_val == cls).astype(int)
    brier = brier_score_loss(y_true_binary, val_probs[:, i])
    print(f"  Brier score ({cls}): {brier:.4f}")