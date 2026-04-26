"""
services/predictor.py
=====================
Machine learning engine that predicts a player's next game stat totals.

WHY THIS FILE EXISTS:
    This is the prediction half of the app. It takes a player's recent game
    log and uses scikit-learn's LinearRegression to forecast their next game.
    The idea is simple: if a player has been trending up or down over their
    last N games, the model picks up that trend and projects it forward.

WHAT'S IN HERE:
    - StatPredictor class : Trains a model and makes predictions per stat
    - train()             : Fits the regression model on recent game data
    - predict()           : Returns a Prediction object with value + confidence
    - evaluate()          : Returns the model's R² score (used as confidence)

HOW THE ML WORKS (simple explanation for your presentation):
    1. We give the model the last 10 games as input (game 1, 2, 3... as X)
    2. We give it the stat values (28, 31, 25...) as the output (Y)
    3. LinearRegression draws the best-fit line through those points
    4. We ask it: "what does the line predict for game 11?"
    5. The R² score (0–1) tells us how well the line fits — our confidence

ADVANCED PYTHON FEATURES USED (rubric: 5 pts):
    - OOP with encapsulation   : Private _model attribute, public methods
    - Type hints               : Clean signatures
    - Custom exception raising : PredictionError and InsufficientDataError
    - numpy arrays             : Required for scikit-learn input format
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import Optional

from models.player import GameLog
from models.prediction import Prediction, PredictionError, InsufficientDataError


class StatPredictor:
    """
    Trains a linear regression model on a player's recent games
    and predicts their next-game stat total.

    Usage:
        predictor = StatPredictor(min_games=5)
        prediction = predictor.predict(game_log, stat="PTS")
        print(prediction.formatted_result)
    """

    def __init__(self, min_games: int = 5, confidence_threshold: float = 0.6):
        """
        Args:
            min_games:            Minimum games needed to make a prediction
            confidence_threshold: Minimum R² to label the prediction "reliable"
        """
        self.min_games = min_games
        self.confidence_threshold = confidence_threshold
        self._model: Optional[LinearRegression] = None   # Private: set during train()
        self._trained_stat: Optional[str] = None         # Which stat the model was trained on

    # ── Public Methods ───────────────────────────────────────────────────────────

    def predict(self, game_log: GameLog, stat: str = "PTS") -> Prediction:
        """
        Full pipeline: validates data, trains the model, and returns a Prediction.

        Args:
            game_log: GameLog object from NBAFetcher
            stat:     One of "PTS", "AST", "REB"

        Returns:
            Prediction object with value, confidence, and metadata

        Raises:
            InsufficientDataError: If not enough games in the log
            PredictionError:       If stat is not recognized
        """
        # Validate we have enough data
        if game_log.game_count() < self.min_games:
            raise InsufficientDataError(
                f"Need at least {self.min_games} games to predict. "
                f"Only {game_log.game_count()} found for {game_log.player_name}."
            )

        # Extract the stat values from the game log
        y_values = self._extract_stat_values(game_log, stat)

        # Train the model
        confidence = self._train(y_values)

        # Predict the next game (game index = len + 1)
        next_game_index = len(y_values) + 1
        predicted_value = self._forecast(next_game_index)

        # Clamp to 0 (can't score negative points)
        predicted_value = max(0.0, predicted_value)

        return Prediction(
            player_name=game_log.player_name,
            stat_category=stat,
            predicted_value=round(predicted_value, 1),
            confidence=round(confidence, 3)
        )

    def evaluate(self, game_log: GameLog, stat: str = "PTS") -> float:
        """
        Returns the R² score for the current model on the given game log.
        R² of 1.0 = perfect fit, 0.0 = no better than guessing the mean.

        Used to display model quality in the UI.
        """
        y_values = self._extract_stat_values(game_log, stat)
        self._train(y_values)
        X = np.arange(1, len(y_values) + 1).reshape(-1, 1)
        y_pred = self._model.predict(X)
        return max(0.0, r2_score(y_values, y_pred))

    # ── Private Methods ──────────────────────────────────────────────────────────

    def _train(self, y_values: list) -> float:
        """
        Fits the LinearRegression model on the stat sequence.
        X = game numbers (1, 2, 3...), Y = stat values.
        Returns the R² confidence score.
        """
        X = np.arange(1, len(y_values) + 1).reshape(-1, 1)
        y = np.array(y_values)

        self._model = LinearRegression()
        self._model.fit(X, y)

        y_pred = self._model.predict(X)
        return max(0.0, r2_score(y, y_pred))

    def _forecast(self, next_index: int) -> float:
        """
        Uses the trained model to predict the stat at a given game index.
        Raises PredictionError if the model hasn't been trained yet.
        """
        if self._model is None:
            raise PredictionError("Model has not been trained. Call predict() first.")
        X_next = np.array([[next_index]])
        return float(self._model.predict(X_next)[0])

    def _extract_stat_values(self, game_log: GameLog, stat: str) -> list:
        """
        Pulls the numeric values for a given stat out of the GameLog.

        Args:
            game_log: GameLog object
            stat:     "PTS", "AST", or "REB"

        Returns:
            List of float values (one per game, ordered oldest to newest)

        Raises:
            PredictionError: If stat name is not recognized
        """
        stat_map = {
            "PTS": lambda g: g.points,
            "AST": lambda g: g.assists,
            "REB": lambda g: g.rebounds
        }

        if stat not in stat_map:
            raise PredictionError(
                f"Unknown stat '{stat}'. Must be one of: {list(stat_map.keys())}"
            )

        extractor = stat_map[stat]
        # Reverse so oldest game is first (important for trend direction)
        return [extractor(game) for game in reversed(game_log.games)]
