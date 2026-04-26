"""
models/player.py
================
Defines the core data structures for a Player and their stats.

WHY THIS FILE EXISTS:
    Instead of passing around messy dictionaries like {"name": "LeBron", "pts": 28},
    we use Python dataclasses to create clean, structured objects with built-in
    validation. This is the OOP requirement for your rubric.

WHAT'S IN HERE:
    - Player       : Holds a player's identity info (name, team, position)
    - PlayerStats  : Holds a player's stat line for a single game or season average
    - GameLog      : Holds a list of recent games for a player

ADVANCED PYTHON FEATURES USED (rubric: 5 pts):
    - @dataclass decorator       : Auto-generates __init__, __repr__, __eq__
    - @property decorator        : Computed attributes with clean syntax
    - __post_init__              : Runs validation right after object is created
    - Type hints throughout      : Makes code readable and catches bugs early
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Player:
    """
    Represents an NBA player's identity.

    Example usage:
        player = Player(name="LeBron James", team="Los Angeles Lakers", position="SF")
        print(player.display_name)  # "LeBron James (SF) — Los Angeles Lakers"
    """
    name: str
    team: str
    position: str
    player_id: Optional[int] = None     # NBA API's internal player ID

    def __post_init__(self):
        """Runs automatically after __init__. Validates and cleans input."""
        if not self.name or not self.name.strip():
            raise ValueError("Player name cannot be empty.")
        self.name = self.name.strip().title()   # Normalize: "lebron james" → "Lebron James"
        self.team = self.team.strip()

    @property
    def display_name(self) -> str:
        """Returns a nicely formatted string for display in the UI."""
        return f"{self.name} ({self.position}) — {self.team}"


@dataclass
class PlayerStats:
    """
    Represents a player's stat line — either a single game or a season average.

    Example usage:
        stats = PlayerStats(player_name="LeBron James", points=28.5, assists=7.2, rebounds=7.5)
        print(stats.summary)  # "28.5 PTS | 7.2 AST | 7.5 REB | 0.0% FG"
    """
    player_name: str
    points: float = 0.0
    assists: float = 0.0
    rebounds: float = 0.0
    fg_percentage: float = 0.0          # Field goal % as decimal (e.g. 0.52 = 52%)
    fg3_percentage: float = 0.0         # 3-point % as decimal
    games_played: int = 0
    is_season_average: bool = False     # True = averages, False = single game

    def __post_init__(self):
        """Validates that stat values are non-negative numbers."""
        for stat_name, value in [("points", self.points),
                                  ("assists", self.assists),
                                  ("rebounds", self.rebounds)]:
            if value < 0:
                raise ValueError(f"{stat_name} cannot be negative. Got: {value}")

    @property
    def summary(self) -> str:
        """One-line summary for display in the UI."""
        return (f"{self.points:.1f} PTS | {self.assists:.1f} AST | "
                f"{self.rebounds:.1f} REB | {self.fg_percentage * 100:.1f}% FG")


@dataclass
class GameLog:
    """
    Holds a list of recent games for a player — used to feed the ML predictor.

    Example usage:
        log = GameLog(player_name="LeBron James", games=[stats1, stats2, stats3])
        print(log.average_points)  # Average points across all games in the log
    """
    player_name: str
    games: List[PlayerStats] = field(default_factory=list)  # default_factory avoids mutable default bug

    @property
    def average_points(self) -> float:
        """Average points per game across the game log."""
        if not self.games:
            return 0.0
        return sum(g.points for g in self.games) / len(self.games)

    @property
    def average_assists(self) -> float:
        """Average assists per game across the game log."""
        if not self.games:
            return 0.0
        return sum(g.assists for g in self.games) / len(self.games)

    @property
    def average_rebounds(self) -> float:
        """Average rebounds per game across the game log."""
        if not self.games:
            return 0.0
        return sum(g.rebounds for g in self.games) / len(self.games)

    def game_count(self) -> int:
        """Returns how many games are in this log."""
        return len(self.games)
