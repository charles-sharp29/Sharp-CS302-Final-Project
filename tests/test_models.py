"""
tests/test_models.py
====================
Unit tests for the core data model classes.

These tests validate pure-Python logic only (no API calls, no network).
"""

import pytest

from models.player import GameLog, Player, PlayerStats
from models.prediction import Prediction, PredictionError


class TestPlayer:
    """Tests for the Player dataclass."""

    def test_valid_player_creation(self) -> None:
        """A Player with valid inputs should be created successfully."""
        player = Player(name="LeBron James", team="Los Angeles Lakers", position="SF")
        assert player.name == "Lebron James"
        assert player.team == "Los Angeles Lakers"
        assert player.position == "SF"
        assert player.player_id is None

    def test_player_name_normalization_strip_title(self) -> None:
        """Player name should be normalized using .strip().title()."""
        player = Player(name="  lebron james  ", team="Lakers", position="SF")
        assert player.name == "Lebron James"

    def test_player_empty_name_raises_value_error(self) -> None:
        """Empty (or whitespace-only) player name should raise ValueError."""
        with pytest.raises(ValueError):
            Player(name="", team="Lakers", position="SF")

        with pytest.raises(ValueError):
            Player(name="   ", team="Lakers", position="SF")

    def test_player_display_name_format(self) -> None:
        """display_name should follow the '{name} ({position}) — {team}' format."""
        player = Player(name="Stephen Curry", team="Golden State Warriors", position="PG")
        assert player.display_name == "Stephen Curry (PG) — Golden State Warriors"


class TestPlayerStats:
    """Tests for the PlayerStats dataclass."""

    def test_valid_stats_creation(self) -> None:
        """PlayerStats with non-negative values should be created successfully."""
        stats = PlayerStats(player_name="LeBron James", points=28.5, assists=7.2, rebounds=7.5)
        assert stats.points == pytest.approx(28.5)
        assert stats.assists == pytest.approx(7.2)
        assert stats.rebounds == pytest.approx(7.5)

    def test_summary_includes_pts_ast_reb(self) -> None:
        """summary should include formatted PTS/AST/REB tokens."""
        stats = PlayerStats(player_name="Test", points=25.0, assists=5.0, rebounds=8.0)
        summary = stats.summary
        assert "25.0 PTS" in summary
        assert "5.0 AST" in summary
        assert "8.0 REB" in summary

    def test_negative_points_raises_value_error(self) -> None:
        """Negative points should raise ValueError."""
        with pytest.raises(ValueError):
            PlayerStats(player_name="Test", points=-1.0)

    def test_zero_stats_are_valid(self) -> None:
        """Zero values should be allowed (e.g., DNP or low-minute games)."""
        stats = PlayerStats(player_name="Test", points=0.0, assists=0.0, rebounds=0.0)
        assert stats.points == 0.0
        assert stats.assists == 0.0
        assert stats.rebounds == 0.0


class TestGameLog:
    """Tests for the GameLog dataclass."""

    @staticmethod
    def _make_log() -> GameLog:
        """Create a small GameLog with 3 games for testing."""
        games = [
            PlayerStats(player_name="Test", points=20.0, assists=5.0, rebounds=6.0),
            PlayerStats(player_name="Test", points=30.0, assists=7.0, rebounds=8.0),
            PlayerStats(player_name="Test", points=25.0, assists=6.0, rebounds=7.0),
        ]
        return GameLog(player_name="Test Player", games=games)

    def test_average_points_returns_correct_mean(self) -> None:
        """average_points should compute the mean across all games."""
        log = self._make_log()
        assert log.average_points == pytest.approx(25.0)

    def test_empty_log_returns_zero(self) -> None:
        """Empty logs should return 0.0 for all averages."""
        log = GameLog(player_name="Empty Player", games=[])
        assert log.average_points == 0.0
        assert log.average_assists == 0.0
        assert log.average_rebounds == 0.0

    def test_game_count_returns_correct_number(self) -> None:
        """game_count() should match the number of stored games."""
        log = self._make_log()
        assert log.game_count() == 3


class TestPrediction:
    """Tests for the Prediction dataclass and exceptions."""

    def test_valid_prediction_creation(self) -> None:
        """A valid Prediction should be created successfully."""
        pred = Prediction(
            player_name="LeBron James", stat_category="PTS", predicted_value=27.5
        )
        assert pred.predicted_value == pytest.approx(27.5)
        assert pred.stat_category == "PTS"

    def test_negative_predicted_value_raises_prediction_error(self) -> None:
        """Negative predicted_value should raise PredictionError."""
        with pytest.raises(PredictionError):
            Prediction("Test", "PTS", predicted_value=-5.0)

    def test_to_dict_has_required_keys(self) -> None:
        """to_dict() should include all keys needed for JSON storage."""
        pred = Prediction("LeBron James", "PTS", 27.5)
        result = pred.to_dict()
        required_keys = {
            "player_name",
            "stat_category",
            "predicted_value",
            "ai_insight",
            "created_at",
        }
        assert required_keys.issubset(result.keys())

    def test_formatted_result_contains_player_name(self) -> None:
        """formatted_result should contain the player's name."""
        pred = Prediction("Steph Curry", "PTS", 31.0)
        assert "Steph Curry" in pred.formatted_result

