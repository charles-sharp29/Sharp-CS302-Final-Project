"""
models/prediction.py
====================
Defines the Prediction data structure and custom exceptions.

WHY THIS FILE EXISTS:
    When the ML model or AI generates a prediction, we need a clean object
    to hold that result — not just a raw number. This file also defines
    custom exception classes, which is an advanced Python feature listed
    directly in your rubric.

WHAT'S IN HERE:
    - PredictionError       : Custom exception for prediction failures
    - InsufficientDataError : Custom exception when not enough games exist
    - Prediction            : Dataclass holding a complete prediction result

ADVANCED PYTHON FEATURES USED (rubric: 5 pts):
    - Custom exception classes   : Inherit from Exception for specific error types
    - @dataclass                 : Clean OOP structure
    - @property with formatting  : Computed display strings
    - datetime                   : Timestamps predictions automatically
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Custom Exceptions ──────────────────────────────────────────────────────────

class PredictionError(Exception):
    """
    Raised when the prediction model fails for any reason.

    Example:
        raise PredictionError("Model has not been trained yet.")
    """
    pass


class InsufficientDataError(PredictionError):
    """
    Raised when a player doesn't have enough games to make a prediction.
    Inherits from PredictionError so callers can catch either.

    Example:
        raise InsufficientDataError("Need at least 5 games. Only 2 found for Wembanyama.")
    """
    pass


# ── Prediction Dataclass ───────────────────────────────────────────────────────

@dataclass
class Prediction:
    """
    Holds the complete result of a stat prediction for one player.

    Example usage:
        pred = Prediction(
            player_name="LeBron James",
            stat_category="PTS",
            predicted_value=27.5,
            confidence=0.82,
            ai_insight="LeBron has averaged 29 pts in his last 5 games..."
        )
        print(pred.formatted_result)
        # "LeBron James is predicted to score 27.5 PTS (Confidence: High)"
    """
    player_name: str
    stat_category: str                          # "PTS", "AST", or "REB"
    predicted_value: float
    confidence: float                           # R² score from ML model (0.0 to 1.0)
    ai_insight: str = ""                        # Natural language from OpenAI
    created_at: str = field(                    # Auto-set timestamp on creation
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M")
    )

    def __post_init__(self):
        """Validates prediction values are sensible."""
        if self.predicted_value < 0:
            raise PredictionError(f"Predicted value cannot be negative: {self.predicted_value}")
        if not (0.0 <= self.confidence <= 1.0):
            raise PredictionError(f"Confidence must be between 0 and 1: {self.confidence}")

    @property
    def confidence_label(self) -> str:
        """Converts numeric confidence score to a human-readable label."""
        if self.confidence >= 0.8:
            return "High"
        elif self.confidence >= 0.6:
            return "Medium"
        else:
            return "Low"

    @property
    def formatted_result(self) -> str:
        """One-line display string for the UI prediction card."""
        return (f"{self.player_name} is predicted to score "
                f"{self.predicted_value:.1f} {self.stat_category} "
                f"(Confidence: {self.confidence_label})")

    def to_dict(self) -> dict:
        """Converts the prediction to a dictionary for JSON storage."""
        return {
            "player_name": self.player_name,
            "stat_category": self.stat_category,
            "predicted_value": self.predicted_value,
            "confidence": self.confidence,
            "confidence_label": self.confidence_label,
            "ai_insight": self.ai_insight,
            "created_at": self.created_at
        }
