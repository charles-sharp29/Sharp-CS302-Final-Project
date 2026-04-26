"""
models/player.py
================
Core data structures for the NBA analytics app.

This module defines simple, validated dataclasses used across the app:
- Player: identity information for a player
- PlayerStats: a single stat line (game or season average)
- GameLog: a collection of PlayerStats with computed averages
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Player:
    """
    Represents a single NBA player's identity information.

    This class is intentionally small and immutable-by-convention; it standardizes
    player naming and provides a consistent UI-ready display string.
    """

    name: str
    team: str
    position: str
    player_id: Optional[int] = None
    team_abbreviation: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Validate and normalize the player fields after initialization.

        - Ensures the name is not empty.
        - Normalizes the name to stripped title-case for consistent display.
        """

        if not self.name or not self.name.strip():
            raise ValueError("Player name cannot be empty.")
        self.name = self.name.strip().title()

    @property
    def display_name(self) -> str:
        """
        Return a friendly, UI-ready representation of the player.

        Format: "{name} ({position}) — {team}"
        """

        return f"{self.name} ({self.position}) — {self.team}"

    @property
    def headshot_url(self) -> Optional[str]:
        """
        Return the URL of the player's official NBA headshot.

        Uses the public NBA CDN at the 260x190 size (same resolution NBA.com
        itself uses on player-card widgets). Returns None if we don't have a
        player_id to build the URL with.
        """

        if self.player_id is None:
            return None
        return f"https://cdn.nba.com/headshots/nba/latest/260x190/{self.player_id}.png"


@dataclass
class PlayerStats:
    """
    Represents a player's statistical line for a single game or a season average.

    All numeric stats are validated to be non-negative.
    """

    player_name: str
    points: float = 0.0
    assists: float = 0.0
    rebounds: float = 0.0
    fg_percentage: float = 0.0
    fg3_percentage: float = 0.0
    games_played: int = 0
    is_season_average: bool = False
    game_date: Optional[str] = None
    opponent_abbreviation: Optional[str] = None
    # True when this stat line came from a postseason (Playoffs) game. The
    # NBA's `PlayerGameLog` endpoint returns regular season and playoff
    # rows separately; the fetcher tags merged rows so the dashboard can
    # flag playoff games and the predictor can weigh them appropriately.
    is_playoff_game: bool = False

    def __post_init__(self) -> None:
        """
        Validate stats after initialization.

        Raises:
            ValueError: If any numeric statistic is negative.
        """

        float_fields: List[tuple[str, float]] = [
            ("points", self.points),
            ("assists", self.assists),
            ("rebounds", self.rebounds),
            ("fg_percentage", self.fg_percentage),
            ("fg3_percentage", self.fg3_percentage),
        ]
        for field_name, value in float_fields:
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative. Got: {value}")

        if self.games_played < 0:
            raise ValueError(f"games_played cannot be negative. Got: {self.games_played}")

    @property
    def summary(self) -> str:
        """
        Return a compact, human-readable summary of the stat line.

        The string is intended for display in the Streamlit UI.
        """

        base = (
            f"{self.points:.1f} PTS | {self.assists:.1f} AST | {self.rebounds:.1f} REB"
            f" | FG {self.fg_percentage * 100.0:.1f}% | 3P {self.fg3_percentage * 100.0:.1f}% | GP {self.games_played}"
        )
        if self.is_season_average:
            return f"{base} | Season Avg"
        return base


@dataclass
class GameLog:
    """
    Represents a collection of games (PlayerStats) for a given player.

    This is typically used to compute rolling averages for analytics/prediction.
    """

    player_name: str
    games: List[PlayerStats] = field(default_factory=list)

    @property
    def average_points(self) -> float:
        """
        Return the average points across all games in the log.

        Returns 0.0 when there are no games.
        """

        if not self.games:
            return 0.0
        return sum(g.points for g in self.games) / len(self.games)

    @property
    def average_assists(self) -> float:
        """
        Return the average assists across all games in the log.

        Returns 0.0 when there are no games.
        """

        if not self.games:
            return 0.0
        return sum(g.assists for g in self.games) / len(self.games)

    @property
    def average_rebounds(self) -> float:
        """
        Return the average rebounds across all games in the log.

        Returns 0.0 when there are no games.
        """

        if not self.games:
            return 0.0
        return sum(g.rebounds for g in self.games) / len(self.games)

    def game_count(self) -> int:
        """
        Return the number of games currently stored in this log.
        """

        return len(self.games)

