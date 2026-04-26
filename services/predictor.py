"""
services/predictor.py
=====================
Machine learning prediction engine for forecasting a player's next-game stats.

This module uses a Random Forest Regressor (not linear regression) because the
relationships we care about are non-linear. For example, the effect of rest
days on performance is not a straight line — 0 days of rest hurts scoring
(fatigue), 1–2 days tends to be ideal, and 3+ days can show rust. Tree-based
models handle those kinks naturally.

Features used for every game in the log:
    - game_index         : chronological position (1 = oldest)
    - rest_days          : ACTUAL days of rest before that game
                           (gap between this game's date and the previous
                           game's date). The last training row uses the real
                           upcoming-game rest days; intermediate rows use
                           historical gaps. This makes the feature vary, so
                           the forest can actually learn from it.
    - opponent_def_rating: ACTUAL opponent defensive rating for each historical
                           game (looked up in `opp_def_ratings_map` using the
                           opponent tricode parsed from the game log). Varies
                           across rows, so the forest can split on it.
    - rolling_avg_3      : trailing 3-game average of the stat
    - rolling_avg_5      : trailing 5-game average of the stat

The target (y) is the stat value for each game. The model then forecasts a
single synthetic "next-game" row built from the same features, using the real
upcoming opponent's DEF_RATING and the real rest days before the next game.
"""

from __future__ import annotations

from datetime import datetime
from statistics import median
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from models.player import GameLog, PlayerStats
from models.prediction import InsufficientDataError, Prediction, PredictionError


# Bounds for the rest-days feature; anything outside is clamped so freak
# gaps (all-star break, injuries) don't dominate the training signal.
_MIN_REST_DAYS: int = 0
_MAX_REST_DAYS: int = 14


def _parse_nba_date(raw: Optional[str]) -> Optional[datetime]:
    """
    Parse the raw `GAME_DATE` string coming from nba_api's PlayerGameLog.

    nba_api returns dates like "APR 10, 2025". We try that format first,
    then fall back to a few common ISO-style layouts so tests and alternate
    data sources don't have to reformat anything.

    Args:
        raw: Raw date string or None.

    Returns:
        A parsed `datetime` or None if parsing fails.
    """

    if not raw:
        return None
    for fmt in ("%b %d, %Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(raw).strip(), fmt)
        except ValueError:
            continue
    return None


class StatPredictor:
    """
    Trains a Random Forest Regressor on recent games and predicts the next game.

    The predictor is stat-specific per training run (PTS/AST/REB). Rest days
    and opponent defensive rating are per-row historical values so the forest
    actually learns their effect, rather than seeing them as constants.
    """

    def __init__(self, min_games: int = 5, confidence_threshold: float = 0.6) -> None:
        """
        Initialize the predictor with a minimum-games guardrail.

        Args:
            min_games: Minimum number of games required to train/predict.
            confidence_threshold: Unused legacy parameter kept for backward
                compatibility with existing config; no longer surfaced in
                predictions since the R²/confidence output was removed.
        """

        self.min_games = min_games
        self.confidence_threshold = confidence_threshold
        # Fresh Random Forest — n_estimators=100 is a solid default for tabular
        # data this small; random_state=42 keeps training deterministic.
        self._model: RandomForestRegressor = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        self._is_trained: bool = False

    def predict(
        self,
        game_log: GameLog,
        stat: str = "PTS",
        next_rest_days: int = 1,
        next_opponent_def_rating: float = 112.0,
        opp_def_ratings_map: Optional[Dict[str, float]] = None,
    ) -> Prediction:
        """
        Predict the next-game value for a given stat using the recent game log.

        The caller is expected to supply the REAL next-game rest days, the
        REAL next-game opponent DEF_RATING, and a league-wide tricode→DEF_RATING
        map so each historical game gets its actual opponent's defense score.
        When the map is missing, the predictor falls back to the next-game
        opponent's DEF_RATING for historical rows, which is still more honest
        than a hard-coded league average.

        Args:
            game_log: Player game history container.
            stat: Stat category to predict ("PTS", "AST", or "REB").
            next_rest_days: Rest days before the upcoming game (0 = back-to-back).
            next_opponent_def_rating: DEF_RATING of the upcoming opponent.
            opp_def_ratings_map: Optional map of {team_tricode: DEF_RATING} used
                to label each historical game's opponent. Pass the output of
                `NBAFetcher.get_all_team_def_ratings()`.

        Returns:
            A `Prediction` containing the predicted value and confidence score.

        Raises:
            InsufficientDataError: If fewer than `min_games` games are available.
            PredictionError: If an unknown stat category is provided.
        """

        if game_log.game_count() < self.min_games:
            raise InsufficientDataError(
                f"Need at least {self.min_games} games to predict; got {game_log.game_count()}."
            )

        stat_key = stat.strip().upper()

        X, y = self._build_feature_matrix(
            game_log,
            stat_key,
            next_rest_days=next_rest_days,
            next_opponent_def_rating=next_opponent_def_rating,
            opp_def_ratings_map=opp_def_ratings_map or {},
        )
        x_next = self._build_prediction_features(
            game_log,
            stat_key,
            next_rest_days=next_rest_days,
            next_opponent_def_rating=next_opponent_def_rating,
        )

        # Fit and predict. No R² / cross-val score is computed — on 10-game
        # samples it was always ~0 and misleading to expose in the UI.
        self._train(X, y)
        predicted_value = float(self._model.predict(x_next.reshape(1, -1))[0])
        # Clamp to >= 0 so we never report a negative stat value.
        predicted_value = max(0.0, predicted_value)

        return Prediction(
            player_name=game_log.player_name,
            stat_category=stat_key,
            predicted_value=predicted_value,
        )

    def _build_feature_matrix(
        self,
        game_log: GameLog,
        stat: str,
        next_rest_days: int,
        next_opponent_def_rating: float,
        opp_def_ratings_map: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the training feature matrix (X) and target vector (y).

        Unlike the earlier version of this method, `rest_days` and
        `opponent_def_rating` are now REAL historical per-game values: they
        are computed from consecutive `GAME_DATE`s and looked up in the
        league-wide DEF_RATING map. That variation is what lets the forest
        split on these features at all.

        Args:
            game_log: Player game history container.
            stat: Stat category ("PTS", "AST", or "REB").
            next_rest_days: Upcoming rest-days value — used as fallback for
                the chronologically first game (which has no prior game).
            next_opponent_def_rating: Upcoming DEF_RATING — used as fallback
                whenever a historical opponent is missing from the map.
            opp_def_ratings_map: {team_tricode: DEF_RATING} lookup.

        Returns:
            (X, y) where X shape = (N, 5) and y shape = (N,).

        Raises:
            PredictionError: If stat is not recognized.
        """

        # Validate stat (raises PredictionError on bad input) and also return
        # the chronological order — oldest first.
        _ = self._extract_stat_values(game_log, stat)

        # We need full PlayerStats objects (not just the scalar stat) so we
        # can read per-game date + opponent. Reverse to match oldest-first.
        ordered_games: List[PlayerStats] = list(reversed(game_log.games))
        num_games: int = len(ordered_games)

        # Extract (potentially None) per-game dates and opponents, plus the
        # raw stat values for rolling averages.
        dates: List[Optional[datetime]] = [
            _parse_nba_date(g.game_date) for g in ordered_games
        ]
        opponents: List[Optional[str]] = [
            (g.opponent_abbreviation or "").strip().upper() or None
            for g in ordered_games
        ]
        y_values: List[float] = [
            float(self._get_stat_field(g, stat)) for g in ordered_games
        ]

        # Pre-compute per-game rest days. rest_days[i] = days since game i-1.
        rest_days_per_game: List[int] = self._compute_rest_days(
            dates, fallback=next_rest_days
        )

        # Pre-compute per-game opponent defensive rating using the map.
        opp_def_per_game: List[float] = []
        for opp in opponents:
            if opp and opp in opp_def_ratings_map:
                opp_def_per_game.append(float(opp_def_ratings_map[opp]))
            else:
                # No historical DEF_RATING available (team unknown or map
                # incomplete) — fall back to the REAL upcoming opponent's
                # rating instead of a hard-coded league average.
                opp_def_per_game.append(float(next_opponent_def_rating))

        rows: List[List[float]] = []
        for i in range(num_games):
            game_index: int = i + 1

            # Rolling 3-game trailing average. "First 3 games" don't have
            # enough history, so we fall back to that game's own stat.
            if i < 3:
                rolling_3 = float(y_values[i])
            else:
                rolling_3 = float(np.mean(y_values[i - 3 : i]))

            # Rolling 5-game trailing average with the same fallback policy.
            if i < 5:
                rolling_5 = float(y_values[i])
            else:
                rolling_5 = float(np.mean(y_values[i - 5 : i]))

            rows.append(
                [
                    float(game_index),
                    float(rest_days_per_game[i]),
                    float(opp_def_per_game[i]),
                    rolling_3,
                    rolling_5,
                ]
            )

        X = np.array(rows, dtype=float)
        y = np.array(y_values, dtype=float)
        return X, y

    def _build_prediction_features(
        self,
        game_log: GameLog,
        stat: str,
        next_rest_days: int,
        next_opponent_def_rating: float,
    ) -> np.ndarray:
        """
        Build the single feature row used to forecast the upcoming game.

        Args:
            game_log: Player game history container.
            stat: Stat category ("PTS", "AST", or "REB").
            next_rest_days: Rest days before the upcoming game.
            next_opponent_def_rating: Opponent DEF_RATING for the upcoming game.

        Returns:
            A 1-D numpy array of shape (5,) matching the training matrix columns.

        Raises:
            PredictionError: If stat is not recognized.
        """

        y_values: List[float] = self._extract_stat_values(game_log, stat)
        num_games: int = len(y_values)

        # Next game's index extends the chronological sequence by one.
        game_index: int = num_games + 1

        # Rolling_3 = mean of the most recent 3 games (or all if fewer).
        last_3 = y_values[-3:] if num_games >= 3 else y_values
        rolling_3 = float(np.mean(last_3)) if last_3 else 0.0

        # Rolling_5 = mean of the most recent 5 games (or all if fewer).
        last_5 = y_values[-5:] if num_games >= 5 else y_values
        rolling_5 = float(np.mean(last_5)) if last_5 else 0.0

        # Clamp the incoming rest-days value to the same bounds used during
        # training so the forest sees values from the same distribution.
        clamped_rest = max(_MIN_REST_DAYS, min(_MAX_REST_DAYS, int(next_rest_days)))

        return np.array(
            [
                float(game_index),
                float(clamped_rest),
                float(next_opponent_def_rating),
                rolling_3,
                rolling_5,
            ],
            dtype=float,
        )

    def _compute_rest_days(
        self,
        dates: List[Optional[datetime]],
        fallback: int,
    ) -> List[int]:
        """
        Compute historical rest days for each game from their dates.

        Uses the same piecewise rule as `NBAFetcher.get_rest_days` so the
        training features line up with the upcoming-game feature:

            gap = days(i) - days(i-1)
              - gap == 1 (back-to-back)           -> 0 rest days
              - gap == 2 (Mon->Wed)               -> 1 rest day
              - gap >= 3                          -> gap - 2 rest days
                (one day is treated as travel / shoot-around)

        For the first game, or any game missing a usable date, we use the
        median of the other computed values; if nothing can be computed, we
        use the `fallback`.

        Args:
            dates: Per-game parsed datetimes (oldest first); may contain None.
            fallback: Value to use when no historical rest can be derived.

        Returns:
            A list of integers (one per game) in the range [0, 14].
        """

        num_games = len(dates)
        raw: List[Optional[int]] = [None] * num_games

        # Pass 1: compute rest-days where both consecutive dates are present.
        for i in range(1, num_games):
            prev_dt, curr_dt = dates[i - 1], dates[i]
            if prev_dt is not None and curr_dt is not None:
                gap = (curr_dt - prev_dt).days
                # Piecewise rule kept in sync with NBAFetcher.get_rest_days.
                delta = gap - 1 if gap <= 2 else gap - 2
                raw[i] = max(_MIN_REST_DAYS, min(_MAX_REST_DAYS, int(delta)))

        # Figure out a sensible fill value: median of what we computed, else
        # the caller's fallback (typically the upcoming-game rest days).
        computed = [v for v in raw if v is not None]
        fill_value: int = int(median(computed)) if computed else int(fallback)
        # Keep the fill inside the same clamp range as the raw values.
        fill_value = max(_MIN_REST_DAYS, min(_MAX_REST_DAYS, fill_value))

        # Pass 2: fill in any Nones with the computed fill_value.
        return [v if v is not None else fill_value for v in raw]

    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Random Forest model on the training data.

        No score is computed — cross-validated R² on ~10 samples was nearly
        always 0 and was misleading when surfaced in the UI, so the predictor
        no longer reports a confidence value.

        Args:
            X: Feature matrix of shape (N, 5).
            y: Target vector of shape (N,).
        """

        self._model.fit(X, y)
        self._is_trained = True

    def _extract_stat_values(self, game_log: GameLog, stat: str) -> List[float]:
        """
        Extract the requested stat values from a GameLog in chronological order.

        Args:
            game_log: Player game history container.
            stat: Stat category ("PTS", "AST", or "REB").

        Returns:
            A list of floats ordered oldest -> newest.

        Raises:
            PredictionError: If stat is not recognized.
        """

        stat_map = {
            "PTS": lambda g: g.points,
            "AST": lambda g: g.assists,
            "REB": lambda g: g.rebounds,
        }
        if stat not in stat_map:
            raise PredictionError("Unknown stat. Expected one of: PTS, AST, REB.")

        extractor = stat_map[stat]
        # nba_api delivers game_log.games newest-first, so reverse for chronology.
        return [float(extractor(game)) for game in reversed(game_log.games)]

    def _get_stat_field(self, game: PlayerStats, stat: str) -> float:
        """
        Return the numeric value of a stat on a single PlayerStats object.

        Args:
            game: A single game's PlayerStats.
            stat: Stat category ("PTS", "AST", or "REB").

        Returns:
            The stat value as a float.

        Raises:
            PredictionError: If stat is not recognized.
        """

        if stat == "PTS":
            return float(game.points)
        if stat == "AST":
            return float(game.assists)
        if stat == "REB":
            return float(game.rebounds)
        raise PredictionError("Unknown stat. Expected one of: PTS, AST, REB.")
