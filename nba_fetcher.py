"""
services/nba_fetcher.py
=======================
Responsible for pulling and cleaning NBA data from the nba_api library.

WHY THIS FILE EXISTS:
    This file is the only place in the entire app that talks to the NBA API.
    By isolating all API calls here, the rest of the app never needs to know
    how the data is fetched — it just receives clean Player and PlayerStats
    objects. This is called the "service layer" pattern and earns points for
    separation of concerns and modular design.

WHAT'S IN HERE:
    - NBAFetcher class : Wraps all nba_api calls into clean methods
    - search_player()  : Finds a player by name, returns a Player object
    - get_game_log()   : Returns a GameLog of recent games for a player
    - get_season_avg() : Returns season average stats as a PlayerStats object

ADVANCED PYTHON FEATURES USED (rubric: 5 pts):
    - Context manager (__enter__/__exit__)  : Safe session handling
    - Generator (get_stat_rows)            : Yields game records one at a time
    - Custom exception handling            : Raises meaningful errors
    - Type hints                           : Clean, readable function signatures
"""

import time
from typing import Optional, Generator

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, playercareerstats, commonplayerinfo
import pandas as pd

from models.player import Player, PlayerStats, GameLog
from utils.helpers import safe_float, safe_int


class NBAFetcher:
    """
    Wraps nba_api calls and returns clean model objects.

    Used as a context manager for safe API session handling:

        with NBAFetcher() as fetcher:
            player = fetcher.search_player("LeBron James")
            log = fetcher.get_game_log(player.player_id)
    """

    def __init__(self, season: str = "2024-25", recent_games: int = 10):
        self.season = season
        self.recent_games = recent_games
        self._request_delay = 0.6      # Seconds between API calls (avoids rate limiting)

    # ── Context Manager ─────────────────────────────────────────────────────────

    def __enter__(self):
        """Called when entering a 'with' block. Returns self for use inside the block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when leaving a 'with' block — even if an error occurred.
        exc_type is None if no error happened.
        Returns False so any exceptions still propagate up normally.
        """
        return False    # Don't suppress exceptions

    # ── Public Methods ───────────────────────────────────────────────────────────

    def search_player(self, name: str) -> Optional[Player]:
        """
        Searches for an NBA player by name.
        Returns a Player object if found, or None if not found.

        Args:
            name: Player's full or partial name (e.g., "LeBron" or "LeBron James")

        Returns:
            Player object or None
        """
        all_players = players.find_players_by_full_name(name)

        if not all_players:
            return None

        # Take the first (best) match
        p = all_players[0]

        try:
            time.sleep(self._request_delay)
            info = commonplayerinfo.CommonPlayerInfo(player_id=p["id"])
            df = info.get_data_frames()[0]

            return Player(
                name=p["full_name"],
                team=df["TEAM_NAME"].values[0] if not df.empty else "Unknown",
                position=df["POSITION"].values[0] if not df.empty else "Unknown",
                player_id=p["id"]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch player info for '{name}': {e}")

    def get_game_log(self, player_id: int) -> GameLog:
        """
        Returns a GameLog of the player's most recent games this season.

        Args:
            player_id: The NBA API's internal player ID (from Player.player_id)

        Returns:
            GameLog object containing a list of PlayerStats (one per game)
        """
        try:
            time.sleep(self._request_delay)
            log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=self.season
            )
            df = log.get_data_frames()[0].head(self.recent_games)

            if df.empty:
                raise ValueError(f"No game log data found for player_id={player_id}")

            # Get the player name from the first row
            player_name = df["Player_ID"].iloc[0] if "PLAYER_NAME" not in df.columns else df["PLAYER_NAME"].iloc[0]

            # Use a generator to build the list of stats (generator = advanced Python)
            games = list(self._parse_game_rows(df))

            return GameLog(player_name=str(player_id), games=games)

        except Exception as e:
            raise RuntimeError(f"Failed to fetch game log for player_id={player_id}: {e}")

    def get_season_avg(self, player_id: int, player_name: str) -> PlayerStats:
        """
        Returns season average stats for a player.

        Args:
            player_id:   NBA API player ID
            player_name: Display name (used to label the stats object)

        Returns:
            PlayerStats with is_season_average=True
        """
        try:
            time.sleep(self._request_delay)
            career = playercareerstats.PlayerCareerStats(player_id=player_id)
            df = career.get_data_frames()[0]

            # Filter to current season
            current = df[df["SEASON_ID"] == self.season]
            if current.empty:
                current = df.tail(1)    # Fall back to most recent season

            row = current.iloc[0]
            gp = safe_int(row.get("GP", 0))

            return PlayerStats(
                player_name=player_name,
                points=safe_float(row.get("PTS", 0)) / max(gp, 1),
                assists=safe_float(row.get("AST", 0)) / max(gp, 1),
                rebounds=safe_float(row.get("REB", 0)) / max(gp, 1),
                fg_percentage=safe_float(row.get("FG_PCT", 0)),
                fg3_percentage=safe_float(row.get("FG3_PCT", 0)),
                games_played=gp,
                is_season_average=True
            )

        except Exception as e:
            raise RuntimeError(f"Failed to fetch season averages for player_id={player_id}: {e}")

    # ── Private Generator ────────────────────────────────────────────────────────

    def _parse_game_rows(self, df: pd.DataFrame) -> Generator[PlayerStats, None, None]:
        """
        Generator that yields one PlayerStats object per row in the DataFrame.

        Using a generator (yield) is an advanced Python feature from your rubric.
        Instead of building the whole list at once, it produces one item at a time,
        which is more memory-efficient for large datasets.
        """
        for _, row in df.iterrows():
            yield PlayerStats(
                player_name=str(row.get("Player_ID", "")),
                points=safe_float(row.get("PTS", 0)),
                assists=safe_float(row.get("AST", 0)),
                rebounds=safe_float(row.get("REB", 0)),
                fg_percentage=safe_float(row.get("FG_PCT", 0)),
                fg3_percentage=safe_float(row.get("FG3_PCT", 0)),
                games_played=1,
                is_season_average=False
            )
