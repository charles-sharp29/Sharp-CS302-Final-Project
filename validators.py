"""
utils/validators.py
===================
Input validation functions that check user input before processing.

WHY THIS FILE EXISTS:
    Every time a user types something into the app, we need to check that
    it's valid before passing it to the API or ML model. Bad input causes
    confusing crashes. This file catches problems early with clear error
    messages. This earns the UI/UX "input validation" points on your rubric.

WHAT'S IN HERE:
    - validate_player_name()  : Checks that a name is a non-empty string
    - validate_stat_category(): Checks that a stat is one of PTS/AST/REB
    - validate_game_count()   : Checks that game count is a positive integer
    - ValidationError         : Custom exception for validation failures
"""


# ── Custom Exception ────────────────────────────────────────────────────────────

class ValidationError(ValueError):
    """
    Raised when user input fails validation.
    Inherits from ValueError so it can be caught generically if needed.
    """
    pass


# ── Validation Functions ─────────────────────────────────────────────────────────

VALID_STATS = {"PTS", "AST", "REB"}
MIN_GAMES = 5
MAX_GAMES = 82      # Max NBA regular season games


def validate_player_name(name: str) -> str:
    """
    Validates and cleans a player name input.

    Args:
        name: Raw string from the user input field

    Returns:
        Cleaned, title-cased name string

    Raises:
        ValidationError: If name is empty or not a string
    """
    if not isinstance(name, str):
        raise ValidationError("Player name must be a text string.")

    name = name.strip()

    if not name:
        raise ValidationError("Player name cannot be empty. Please enter a name.")

    if len(name) < 2:
        raise ValidationError("Player name is too short. Please enter a full name.")

    if any(char.isdigit() for char in name):
        raise ValidationError("Player name should not contain numbers.")

    return name.title()     # Normalize capitalization


def validate_stat_category(stat: str) -> str:
    """
    Validates that the selected stat is one the app supports.

    Args:
        stat: Stat string (e.g., "PTS", "AST", "REB")

    Returns:
        Uppercased, valid stat string

    Raises:
        ValidationError: If stat is not in VALID_STATS
    """
    stat = stat.strip().upper()

    if stat not in VALID_STATS:
        raise ValidationError(
            f"'{stat}' is not a valid stat. Choose from: {', '.join(sorted(VALID_STATS))}"
        )

    return stat


def validate_game_count(count: int) -> int:
    """
    Validates that a requested game count is within reasonable bounds.

    Args:
        count: Number of recent games requested

    Returns:
        The validated count as an integer

    Raises:
        ValidationError: If count is outside the valid range
    """
    try:
        count = int(count)
    except (TypeError, ValueError):
        raise ValidationError("Game count must be a whole number.")

    if count < MIN_GAMES:
        raise ValidationError(
            f"Game count must be at least {MIN_GAMES}. Got: {count}"
        )

    if count > MAX_GAMES:
        raise ValidationError(
            f"Game count cannot exceed {MAX_GAMES} (one full season). Got: {count}"
        )

    return count
