"""
utils/validators.py
===================
Validation helpers for all user inputs.

These functions provide strict, user-friendly validation with consistent error
types (ValidationError) so the Streamlit UI can catch and display clear messages.
"""


class ValidationError(ValueError):
    """
    Raised when user input fails validation.

    Inheriting from ValueError makes it easy to catch either ValidationError
    specifically or value-related errors generically.
    """


VALID_STATS = {"PTS", "AST", "REB"}
MIN_GAMES = 5
MAX_GAMES = 82


def validate_player_name(name: str) -> str:
    """
    Validate and normalize a player name provided by the user.

    Rules:
        - Must be a string
        - Must not be empty after stripping whitespace
        - Must be at least 2 characters long
        - Must not contain any digits

    Args:
        name: Raw player name input.

    Returns:
        The cleaned name as `name.strip().title()`.

    Raises:
        ValidationError: If any rule is violated.
    """

    if not isinstance(name, str):
        raise ValidationError("Player name must be a string.")

    cleaned = name.strip()
    if not cleaned:
        raise ValidationError("Player name cannot be empty.")
    if len(cleaned) < 2:
        raise ValidationError("Player name must be at least 2 characters long.")
    if any(ch.isdigit() for ch in cleaned):
        raise ValidationError("Player name cannot contain digits.")

    return cleaned.title()


def validate_stat_category(stat: str) -> str:
    """
    Validate a stat category selection.

    The input is stripped and uppercased before validation.

    Args:
        stat: Stat category string (e.g., "PTS", "AST", "REB").

    Returns:
        The validated stat category in uppercase.

    Raises:
        ValidationError: If the normalized stat is not one of VALID_STATS.
    """

    normalized = stat.strip().upper()
    if normalized not in VALID_STATS:
        raise ValidationError(f"Invalid stat category '{normalized}'. Valid options: {sorted(VALID_STATS)}")
    return normalized


def validate_game_count(count: int) -> int:
    """
    Validate the number of games requested for analysis.

    The input is converted to int; conversion failure and out-of-range values
    raise ValidationError.

    Args:
        count: Requested number of games (may be int-like).

    Returns:
        The validated integer game count.

    Raises:
        ValidationError: If conversion fails or the value is outside bounds.
    """

    try:
        games = int(count)
    except (TypeError, ValueError):
        raise ValidationError("Game count must be an integer.")

    if games < MIN_GAMES or games > MAX_GAMES:
        raise ValidationError(f"Game count must be between {MIN_GAMES} and {MAX_GAMES}. Got: {games}")

    return games

