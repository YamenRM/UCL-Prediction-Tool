# UCL Prediction Tool ⚽🤖

A machine learning pipeline that predicts UEFA Champions League and Big-5 domestic league match outcomes (Home / Draw / Away), built end-to-end from raw web scraping through a trained, evaluated, and served prediction model.

## Project Status

- ✅ **Data Phase** — Complete
- ✅ **Model Phase** — Complete
- ⬜ **Deploy Phase** — Up next
- ⬜ **DevOps Phase**
- ⬜ **After Development Phase**

---

## 📊 Data Phase — Summary

**Sources**
- [FBref](https://fbref.com) — Big-5 European domestic leagues + UEFA Champions League schedules and results (custom `soccerdata` league config for UCL, since it's not covered by default)
- [ClubElo](http://clubelo.com) — historical Elo ratings, joined per-team via `merge_asof` for leakage-safe "rating as of matchday" lookups
- [SoFIFA](https://sofifa.com) — FIFA/EA FC squad overall ratings across game versions, used as a proxy for transfer-window squad strength changes

**Pipeline stages**
```
Scrape (FBref) → Elo join (ClubElo) → Squad rating join (SoFIFA)
→ Feature engineering (form, rest days, missing-value handling)
→ training_data.csv / to_predict_data.csv
```

**Key engineering challenges resolved**
- **Team name reconciliation**: FBref, ClubElo, and SoFIFA each use different naming conventions for the same clubs (`"Bayern Munich"` vs `"Bayern"` vs `"FC Bayern München"`). Built a `teamname_replacements.json` covering 150+ clubs, combining fuzzy-matching (RapidFuzz) with manual verification against each source's real team lists to avoid false-positive matches.
- **CAPTCHA-corrupted cache bug**: a blocked FBref request silently mislabeled ~460 rows of 1926-era historical data as the current season. Traced to the exact corrupted cache file via scrape timestamps, deleted it, and re-ran ingestion clean.
- **Leakage-safe joins throughout**: every time-based feature (Elo, form, rest days) uses `merge_asof(direction='backward')`, guaranteeing no match is ever scored using information that wasn't actually available before kickoff.

**Final dataset stats**
| Metric | Value |
|---|---|
| Training matches (played, 2021/22–2025/26) | 9,672 |
| Unplayed fixtures (2026/27, live season) | 1,777 |
| Elo coverage | ~99.8% |
| Squad rating coverage | ~91.6% |
| Result distribution (H / A / D) | ~44% / 31% / 25% |

---

## 🧠 Model Phase — Summary

**Feature set** (24 features): Elo differential, home/away Elo, squad rating differential, squad rating delta (transfer-window proxy), rolling form (points + goal difference, 5-match window), rest-day differential, plus missing-data indicator flags.

**Model comparison**

| Model | Log Loss | Accuracy |
|---|---|---|
| Random Forest | **0.9910** | 0.5170 |
| Logistic Regression (full features) | 0.9924 | 0.5118 |
| Logistic Regression (Elo only) | 0.9935 | 0.5170 |
| LightGBM | 0.9937 | 0.5134 |
| XGBoost | 0.9939 | 0.5154 |

All five models landed within ~0.4% log loss of each other — with strong hand-engineered features (Elo, squad ratings), model architecture provided only marginal lift. Random Forest was selected based on its consistent edge (verified stable across 5 random seeds) and lower overfitting risk relative to the boosting models.

**Validation methodology**
- Strict chronological split: train on 2021/22–2024/25, validate on 2025/26 (never seen during training or tuning)
- Recency-weighted training via exponential decay (half-life ≈ 500 days) — recent matches carry up to ~7x the training influence of matches from 2021
- Two separate model artifacts: an evaluation model (excludes 2025/26) for honest metrics, and a production model (trained on all history through 2025/26) for live predictions — preventing validation-season leakage into reported results

**Final evaluation (genuinely unseen 2025/26 season)**
| Metric | Value |
|---|---|
| Accuracy | 51.6% (vs. 44.6% naive baseline) |
| Log loss | 0.991 |
| Draw calibration | Well-calibrated (predicted ~28% draw rate → actual ~29%) |

**Notable finding — the Draw problem**: standard argmax prediction never selected "Draw" as an outcome, despite well-calibrated draw probabilities underneath. Root cause: draws don't have an independent signal region — they emerge when Home/Away probabilities converge, so Draw rarely wins outright. Solved with a margin-based decision rule (if the top two class probabilities are within a threshold of each other, predict Draw) rather than plain argmax, bringing predicted draw rate from 0.1% to a realistic ~18%.

**Top predictive features**: `elo_diff` (26.8%), `squad_rating_diff` (16.6%), `home_elo` (11.6%), `away_elo` (8.8%), `form_gd_diff` (6.6%) — confirming both the Elo signal and the SoFIFA-based transfer-window proxy carry genuine predictive weight.

---

## 🗺️ Roadmap

- **Deploy Phase**: serve predictions via a Streamlit app; surface probability breakdowns and data-quality flags (imputed vs. real Elo/rating data) per fixture
- **DevOps Phase**: orchestrate the six-stage pipeline into a single runnable command; scheduled retraining as the 2026/27 season progresses
- **After Development Phase**: walk-forward backtesting to validate the weekly retrain-and-predict strategy against real historical deployment conditions; potential Transfermarkt-based transfer feature as an upgrade over the SoFIFA proxy

---

## Tech Stack
`Python` · `pandas` · `scikit-learn` · `soccerdata` · `RapidFuzz` · `XGBoost` / `LightGBM` (comparison) · `Streamlit` (planned)

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

`soccerdata` reads two config files from its local data directory — **not committed to this repo**, since they live outside the project folder and are environment-specific. You need to create both before running the scraper, or `INT-Champions League` and several club lookups (ClubElo/SoFIFA) will fail.

**Location**: `~/soccerdata/config/` (Windows: `%USERPROFILE%\soccerdata\config\`). This folder is created automatically the first time you `import soccerdata`, but the two files inside it are not — you add them yourself.

**1. `league_dict.json`** — registers UEFA Champions League as a custom FBref competition (not included by default) `See Importnant/league_dict.json`

**2. `teamname_replacements.json`** — reconciles club naming differences across FBref, ClubElo, and SoFIFA (e.g. `"Bayern Munich"` → `"Bayern"` on ClubElo, `"FC Bayern München"` on SoFIFA). The full mapping (150+ clubs) is maintained separately — see `Important/teamname_replacements.json`.

Once both files are in place, run the pipeline in order:
```bash
python src/data_scraping.py
python src/elo_features.py
python src/transfer_features.py
python src/feature_engineering.py
python src/train.py
python src/predict.py
```
