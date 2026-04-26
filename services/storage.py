"""
services/storage.py
===================
Handles saving and loading JSON data files for persistence.

This module provides a StorageManager class used by the Streamlit app to persist:
- Favorite players
- Prediction history
"""

import json
from pathlib import Path
from typing import Any, List, Optional

from models.prediction import Prediction


class StorageManager:
    """
    Manages JSON file persistence for favorites and predictions.
    """

    def __init__(
        self,
        data_dir: str = "data",
        favorites_file: str = "favorites.json",
        predictions_file: str = "predictions.json",
    ) -> None:
        """
        Initialize storage paths and ensure the data directory exists.

        Args:
            data_dir: Directory where JSON files are stored.
            favorites_file: Filename for favorites JSON.
            predictions_file: Filename for predictions JSON.
        """

        self.data_dir = Path(data_dir)
        self.favorites_path = self.data_dir / favorites_file
        self.predictions_path = self.data_dir / predictions_file

        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_favorite(self, player_dict: dict) -> bool:
        """
        Save a player dictionary to favorites, avoiding duplicates by name.

        Duplicate detection is case-insensitive on the "name" field.

        Args:
            player_dict: Player data (must include a "name" key for de-duplication).

        Returns:
            True if saved, False if a duplicate already exists.
        """

        favorites = self.load_favorites()
        new_name = str(player_dict.get("name", "")).strip().lower()

        existing_names = [str(f.get("name", "")).strip().lower() for f in favorites]
        if new_name and new_name in existing_names:
            return False

        favorites.append(player_dict)
        self._write_json(self.favorites_path, favorites)
        return True

    def load_favorites(self) -> List[dict]:
        """
        Load all favorites from disk.

        Returns:
            A list of favorite player dictionaries, or an empty list if missing.
        """

        data = self._read_json(self.favorites_path, default=[])
        return data if isinstance(data, list) else []

    def remove_favorite(self, player_name: str) -> bool:
        """
        Remove a player from favorites by name (case-insensitive).

        Args:
            player_name: Name of the player to remove.

        Returns:
            True if removed, False if not found.
        """

        favorites = self.load_favorites()
        target = str(player_name).strip().lower()

        filtered = [f for f in favorites if str(f.get("name", "")).strip().lower() != target]

        if len(filtered) == len(favorites):
            return False

        self._write_json(self.favorites_path, filtered)
        return True

    def save_prediction(self, prediction: Prediction) -> None:
        """
        Append a prediction to the predictions history file.

        Args:
            prediction: The Prediction object to save.
        """

        all_predictions = self._read_json(self.predictions_path, default=[])
        if not isinstance(all_predictions, list):
            all_predictions = []
        all_predictions.append(prediction.to_dict())
        self._write_json(self.predictions_path, all_predictions)

    def load_predictions(self, player_name: Optional[str] = None) -> List[dict]:
        """
        Load predictions from disk, optionally filtered by player name.

        Args:
            player_name: If provided, only predictions for this player are returned.

        Returns:
            A list of prediction dictionaries.
        """

        all_predictions = self._read_json(self.predictions_path, default=[])
        if not isinstance(all_predictions, list):
            all_predictions = []

        if player_name is None:
            return all_predictions

        target = str(player_name).strip().lower()
        return [p for p in all_predictions if str(p.get("player_name", "")).strip().lower() == target]

    def _read_json(self, path: Path, default: Any) -> Any:
        """
        Read JSON content from a file path, returning a default on failure.

        Args:
            path: File path to read.
            default: Value returned if the file is missing or invalid JSON.

        Returns:
            Parsed JSON data or the provided default.
        """

        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def _write_json(self, path: Path, data: Any) -> None:
        """
        Write JSON content to a file path.

        Args:
            path: File path to write.
            data: JSON-serializable data to write.
        """

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

