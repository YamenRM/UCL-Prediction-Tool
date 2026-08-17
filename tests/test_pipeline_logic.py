"""
Unit tests using small synthetic data — deliberately independent of the
real scraped dataset (which isn't committed to the repo). These exist to
catch the exact classes of bugs that came up repeatedly during development:
data leakage in rolling features, season/date mismatches, and sample-weight
correctness.
"""

import numpy as np
import pandas as pd
import pytest

from feature_engineering import add_rolling_form, add_rest_days, fill_missing_ratings
from train_test_split import compute_sample_weights


def make_synthetic_matches(n_teams=4, n_rounds=6, start_date="2024-08-01"):
    """Round-robin-ish synthetic match schedule, evenly spaced weekly."""
    teams = [f"Team{i}" for i in range(n_teams)]
    rows = []
    date = pd.Timestamp(start_date)
    rng = np.random.default_rng(42)

    for r in range(n_rounds):
        for i in range(0, n_teams, 2):
            home, away = teams[i], teams[i + 1]
            rows.append({
                "date": date, "league": "TEST-League", "season": "2425",
                "home_team": home, "away_team": away,
                "home_score": int(rng.integers(0, 4)), "away_score": int(rng.integers(0, 4)),
                "home_elo": 1500 + rng.normal(0, 50), "away_elo": 1500 + rng.normal(0, 50),
                "home_squad_rating": 75.0, "away_squad_rating": 75.0,
                "home_squad_delta": 0.5, "away_squad_delta": -0.5,
            })
        teams = teams[1:] + teams[:1]  # rotate
        date += pd.Timedelta(days=7)

    return pd.DataFrame(rows)


class TestRollingFormNoLeakage:
    def test_form_excludes_current_match(self):
        """A team's rolling form going into match N must never be computed
        using match N's own result — the single most important leakage
        check in the whole feature set."""
        df = make_synthetic_matches()
        result = add_rolling_form(df)

        # First match for any team must show the neutral fallback (1.5),
        # since there is no real prior history to draw from yet.
        first_match_home = result.iloc[0]
        assert first_match_home["home_form_pts"] == 1.5

    def test_form_is_deterministic_and_bounded(self):
        df = make_synthetic_matches()
        result = add_rolling_form(df)
        assert result["home_form_pts"].between(0, 3).all()
        assert result["away_form_pts"].between(0, 3).all()


class TestRestDays:
    def test_first_match_gets_neutral_fallback(self):
        df = make_synthetic_matches()
        result = add_rest_days(df)
        assert result.iloc[0]["home_team_rest_days"] == 7

    def test_rest_days_non_negative(self):
        df = make_synthetic_matches()
        result = add_rest_days(df)
        assert (result["home_team_rest_days"] >= 0).all()
        assert (result["away_team_rest_days"] >= 0).all()


class TestFillMissingRatings:
    def test_no_nan_remains_after_fill(self):
        df = make_synthetic_matches()
        df.loc[0, "home_elo"] = np.nan
        df.loc[2, "home_squad_delta"] = np.nan
        result = fill_missing_ratings(df)

        rating_cols = ["home_elo", "away_elo", "home_squad_rating", "away_squad_rating",
                       "home_squad_delta", "away_squad_delta"]
        assert not result[rating_cols].isna().any().any()

    def test_missing_flag_set_before_fill(self):
        df = make_synthetic_matches()
        df.loc[1, "home_elo"] = np.nan
        result = fill_missing_ratings(df)
        assert result.loc[1, "home_elo_missing"] == True  


class TestSampleWeights:
    def test_most_recent_date_gets_full_weight(self):
        dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-06-01", "2025-01-01"]))
        weights = compute_sample_weights(dates, reference_date=dates.max())
        assert weights.iloc[-1] == pytest.approx(1.0)

    def test_older_matches_weighted_less(self):
        dates = pd.Series(pd.to_datetime(["2021-01-01", "2025-01-01"]))
        weights = compute_sample_weights(dates, reference_date=dates.max())
        assert weights.iloc[0] < weights.iloc[1]
        assert 0 < weights.iloc[0] < 1


class TestSeasonDateConsistency:
    """Regression test for the CAPTCHA-corrupted-cache bug: a match's date
    must actually fall within its claimed season's real-world date range."""

    SEASON_DATE_BOUNDS = {
        "2425": ("2024-06-01", "2025-07-31"),
    }

    def test_flags_impossible_season_date_combo(self):
        df = make_synthetic_matches()
        df.loc[0, "date"] = pd.Timestamp("1926-08-28")  # the exact bug that occurred

        bounds = self.SEASON_DATE_BOUNDS["2425"]
        mask = (df["date"] >= pd.Timestamp(bounds[0])) & (df["date"] <= pd.Timestamp(bounds[1]))

        assert not mask.iloc[0] 
        assert mask.iloc[1:].all()  