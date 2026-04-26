"""
models/prediction.py
====================
Prediction data structures and custom exceptions.

This module provides a validated Prediction dataclass used by the app to store
model outputs, display them in the UI, and serialize them for JSON storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


class PredictionError(Exception):
    """
    Base exception for prediction-related validation and runtime errors.

    This is raised when a Prediction instance is created with invalid values or
    when the prediction pipeline encounters a prediction-specific failure.
    """


class InsufficientDataError(PredictionError):
    """
    Raised when there is not enough historical data to produce a prediction.

    This is a specialized PredictionError used to distinguish "not enough data"
    from other prediction validation/runtime failures.
    """


@dataclass
class Prediction:
    """
    Represents a single prediction for a player's stat category.

    Instances are validated on creation and include a convenience property for
    display and a `to_dict()` method for persistence. Confidence / R² scores
    were intentionally removed: on a 10-game sample, a cross-validated R² is
    nearly always 0 and was misleading to the user.
    """

    player_name: str
    stat_category: str
    predicted_value: float
    ai_insight: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

    def __post_init__(self) -> None:
        """
        Validate prediction values after initialization.

        Raises:
            PredictionError: If predicted_value is negative.
        """

        if self.predicted_value < 0:
            raise PredictionError(f"predicted_value must be >= 0. Got: {self.predicted_value}")

    @property
    def formatted_result(self) -> str:
        """
        Return a human-readable sentence summarizing the prediction.
        """

        return (
            f"{self.player_name} — Predicted {self.stat_category}: "
            f"{self.predicted_value:.2f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this prediction into a plain dictionary for JSON storage.

        Returns:
            Dict[str, Any]: All primary fields.
        """

        return {
            "player_name": self.player_name,
            "stat_category": self.stat_category,
            "predicted_value": self.predicted_value,
            "ai_insight": self.ai_insight,
            "created_at": self.created_at,
        }
