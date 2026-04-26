"""
tests/test_services.py
======================
Unit tests for service-layer modules and validators.

These tests cover pure Python logic only:
- StatPredictor (ML wrapper around scikit-learn's Random Forest Regressor)
- StorageManager (JSON persistence using tmp_path)
- Validators (input validation helpers)
"""

from datetime import datetime, timedelta

import pytest

from models.player import GameLog, PlayerStats
from models.prediction import InsufficientDataError, Prediction, PredictionError
from services.predictor import StatPredictor
from services.storage import StorageManager
from utils.validators import (
    ValidationError,
    validate_game_count,
    validate_player_name,
    validate_stat_category,
)


# Rotating set of plausible opponent tricodes so test game logs carry real
# historical context — this is what lets the Random Forest learn per-game
# effects of opponent defense during training.
_OPPONENTS = ["BOS", "MIA", "DEN", "GSW", "NYK", "PHI", "LAL", "HOU", "ATL", "CHI"]


def make_game_log(num_games: int = 10, points: float = 25.0) -> GameLog:
    """
    Create a realistic GameLog for testing.

    Each game increments points by i, while assists/rebounds stay constant.
    Games are dated one day apart (newest-first, matching nba_api's order) and
    carry rotating opponent tricodes so the predictor can compute per-game
    rest days and pick up historical DEF_RATINGs.
    """

    today = datetime(2025, 4, 1)
    games = []
    for i in range(num_games):
        # newest-first: i=0 is the most recent game.
        game_dt = today - timedelta(days=i)
        games.append(
            PlayerStats(
                player_name="Test Player",
                points=points + i,
                assists=5.0,
                rebounds=6.0,
                game_date=game_dt.strftime("%b %d, %Y").upper(),
                opponent_abbreviation=_OPPONENTS[i % len(_OPPONENTS)],
            )
        )
    return GameLog(player_name="Test Player", games=games)


class TestStatPredictor:
    """Tests for StatPredictor behavior and error handling."""

    def test_predict_returns_prediction_object_for_10_game_log(self) -> None:
        """predict() should return a Prediction object for sufficient data."""
        predictor = StatPredictor(min_games=5)
        log = make_game_log(num_games=10)
        result = predictor.predict(
            log,
            stat="PTS",
            next_rest_days=1,
            next_opponent_def_rating=112.0,
            opp_def_ratings_map={},
        )
        assert isinstance(result, Prediction)
        assert result.stat_category == "PTS"

    def test_predicted_value_is_non_negative(self) -> None:
        """predicted_value should always be >= 0."""
        predictor = StatPredictor(min_games=5)
        log = make_game_log(num_games=10, points=20.0)
        result = predictor.predict(
            log,
            stat="PTS",
            next_rest_days=1,
            next_opponent_def_rating=112.0,
            opp_def_ratings_map={},
        )
        assert result.predicted_value >= 0.0

    def test_insufficient_data_error_for_3_game_log(self) -> None:
        """InsufficientDataError should be raised when the log is too short."""
        predictor = StatPredictor(min_games=5)
        log = make_game_log(num_games=3)
        with pytest.raises(InsufficientDataError):
            predictor.predict(
                log,
                stat="PTS",
                next_rest_days=1,
                next_opponent_def_rating=112.0,
            )

    def test_prediction_error_for_unknown_stat(self) -> None:
        """PredictionError should be raised for an invalid stat string."""
        predictor = StatPredictor(min_games=5)
        log = make_game_log(num_games=10)
        with pytest.raises(PredictionError):
            predictor.predict(
                log,
                stat="BLOCKS",
                next_rest_days=1,
                next_opponent_def_rating=112.0,
            )

    def test_predict_works_for_ast(self) -> None:
        """predict() should work for AST as well as PTS."""
        predictor = StatPredictor(min_games=5)
        log = make_game_log(num_games=10)
        result = predictor.predict(
            log,
            stat="AST",
            next_rest_days=1,
            next_opponent_def_rating=112.0,
        )
        assert result.stat_category == "AST"

    def test_high_rest_days_accepted(self) -> None:
        """High rest_days values should be accepted without errors."""
        predictor = StatPredictor(min_games=5)
        log = make_game_log(num_games=10)
        result = predictor.predict(
            log,
            stat="PTS",
            next_rest_days=5,
            next_opponent_def_rating=112.0,
        )
        assert isinstance(result, Prediction)
        assert result.predicted_value >= 0.0

    def test_tough_defense_accepted(self) -> None:
        """Tougher-than-average opponent_def_rating should still return a Prediction."""
        predictor = StatPredictor(min_games=5)
        log = make_game_log(num_games=10)
        result = predictor.predict(
            log,
            stat="PTS",
            next_rest_days=1,
            next_opponent_def_rating=105.0,
        )
        assert isinstance(result, Prediction)
        assert result.predicted_value >= 0.0

    def test_def_rating_map_actually_influences_prediction(self) -> None:
        """The opp_def_ratings map must actually drive predictions.

        Guards against the earlier bug where rest_days / opp_def_rating were
        constant across every training row (same value everywhere) and so the
        Random Forest couldn't split on them. We construct a game log where
        scoring is correlated with opponent defense and verify that (a) two
        identical logs predicted against a soft vs. tough next opponent give
        different numbers, and (b) the "vs. soft defense" number is higher.
        """
        # Historical games: lots of points against soft D, few against tough D.
        opps_with_points = [
            ("BOS", 15.0),
            ("DEN", 16.0),
            ("MIA", 16.5),
            ("GSW", 20.0),
            ("NYK", 22.0),
            ("PHI", 22.5),
            ("LAL", 28.0),
            ("HOU", 30.0),
            ("ATL", 31.0),
            ("CHI", 32.0),
        ]
        today = datetime(2025, 4, 1)
        games = []
        for i, (opp, pts) in enumerate(opps_with_points):
            # Newest-first so nba_api's convention is matched.
            game_dt = today - timedelta(days=i)
            games.append(
                PlayerStats(
                    player_name="Test Player",
                    points=pts,
                    assists=5.0,
                    rebounds=6.0,
                    game_date=game_dt.strftime("%b %d, %Y").upper(),
                    opponent_abbreviation=opp,
                )
            )
        log = GameLog(player_name="Test Player", games=games)

        # Varied DEF_RATINGs that align with the scoring: tough teams have
        # low ratings, soft teams have high ratings.
        def_map = {
            "BOS": 105.0, "DEN": 106.0, "MIA": 107.0,
            "GSW": 112.0, "NYK": 113.0, "PHI": 114.0,
            "LAL": 118.0, "HOU": 119.0, "ATL": 120.0, "CHI": 121.0,
        }

        predictor_a = StatPredictor(min_games=5)
        predictor_b = StatPredictor(min_games=5)
        result_vs_tough = predictor_a.predict(
            log,
            stat="PTS",
            next_rest_days=1,
            next_opponent_def_rating=105.0,
            opp_def_ratings_map=def_map,
        )
        result_vs_soft = predictor_b.predict(
            log,
            stat="PTS",
            next_rest_days=1,
            next_opponent_def_rating=121.0,
            opp_def_ratings_map=def_map,
        )

        # The DEF_RATING feature genuinely varies across rows now, so the
        # forest must produce different predictions, and — given the data —
        # the soft-defense prediction should come in higher.
        assert result_vs_tough.predicted_value != result_vs_soft.predicted_value
        assert result_vs_soft.predicted_value > result_vs_tough.predicted_value


class TestStorageManager:
    """Tests for StorageManager persistence using tmp_path."""

    @pytest.fixture
    def storage(self, tmp_path) -> StorageManager:
        """Create a temporary StorageManager instance for each test."""
        return StorageManager(data_dir=str(tmp_path))

    def test_save_favorite_and_load_favorites(self, storage: StorageManager) -> None:
        """save_favorite() should persist a favorite and load_favorites() should retrieve it."""
        player = {"name": "LeBron James", "team": "Lakers"}
        saved = storage.save_favorite(player)
        assert saved is True
        favorites = storage.load_favorites()
        assert len(favorites) == 1
        assert favorites[0]["name"] == "LeBron James"

    def test_duplicate_save_returns_false_and_no_duplicate_created(self, storage: StorageManager) -> None:
        """Duplicate saves should return False and not create duplicates."""
        player = {"name": "LeBron James", "team": "Lakers"}
        assert storage.save_favorite(player) is True
        assert storage.save_favorite(player) is False
        assert len(storage.load_favorites()) == 1

    def test_remove_favorite_removes_player(self, storage: StorageManager) -> None:
        """remove_favorite() should remove a saved favorite."""
        storage.save_favorite({"name": "LeBron James", "team": "Lakers"})
        removed = storage.remove_favorite("LeBron James")
        assert removed is True
        assert storage.load_favorites() == []

    def test_remove_nonexistent_player_returns_false(self, storage: StorageManager) -> None:
        """Removing a player that does not exist should return False."""
        removed = storage.remove_favorite("Nobody Here")
        assert removed is False

    def test_load_favorites_on_fresh_storage_returns_empty_list(self, storage: StorageManager) -> None:
        """A fresh storage directory should yield no favorites."""
        assert storage.load_favorites() == []


class TestValidators:
    """Tests for input validation helpers."""

    def test_validate_player_name_returns_cleaned_name(self) -> None:
        """Valid names should be stripped and title-cased."""
        assert validate_player_name("  lebron james  ") == "Lebron James"

    def test_validate_player_name_raises_for_empty_string(self) -> None:
        """Empty names should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_player_name("")

    def test_validate_player_name_raises_for_digits(self) -> None:
        """Names containing digits should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_player_name("Player123")

    def test_validate_stat_category_accepts_case_insensitive_inputs(self) -> None:
        """Stat category validation should accept case-insensitive inputs."""
        assert validate_stat_category("pts") == "PTS"
        assert validate_stat_category("AST") == "AST"
        assert validate_stat_category("reb") == "REB"

    def test_validate_stat_category_raises_for_blocks(self) -> None:
        """Unknown stats should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_stat_category("BLOCKS")

    def test_validate_game_count_returns_int_for_valid_input(self) -> None:
        """Valid game count should be returned as an int."""
        assert validate_game_count(10) == 10

    def test_validate_game_count_raises_for_below_min(self) -> None:
        """Counts below minimum should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_game_count(2)

    def test_validate_game_count_raises_for_above_max(self) -> None:
        """Counts above maximum should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_game_count(100)
