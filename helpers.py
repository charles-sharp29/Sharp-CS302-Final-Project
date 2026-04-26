"""
utils/helpers.py
================
Shared utility functions used across the entire app.

WHY THIS FILE EXISTS:
    Small helper functions that multiple files need should live in one place.
    If you need to format a float the same way in three different files,
    you write the function once here and import it everywhere. This is the
    DRY principle (Don't Repeat Yourself) and earns modular design points.

WHAT'S IN HERE:
    - safe_float()       : Converts any value to float safely (no crashes)
    - safe_int()         : Converts any value to int safely
    - format_stat()      : Formats a stat number for clean display
    - load_config()      : Reads and returns the config.yaml file
    - get_stat_color()   : Returns a color string based on stat value (for UI)
"""

import yaml
from pathlib import Path
from typing import Any, Optional


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Converts a value to float without crashing.
    Returns 'default' if conversion fails.

    Example:
        safe_float("28.5")   → 28.5
        safe_float(None)     → 0.0
        safe_float("N/A")    → 0.0
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Converts a value to int without crashing.
    Returns 'default' if conversion fails.

    Example:
        safe_int("42")   → 42
        safe_int(None)   → 0
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def format_stat(value: float, decimal_places: int = 1) -> str:
    """
    Formats a stat number consistently for display in the UI.

    Example:
        format_stat(28.555)  → "28.6"
        format_stat(0.0)     → "0.0"
    """
    return f"{value:.{decimal_places}f}"


def format_percentage(value: float) -> str:
    """
    Formats a decimal percentage for display.

    Example:
        format_percentage(0.523)  → "52.3%"
    """
    return f"{value * 100:.1f}%"


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Reads config.yaml and returns it as a dictionary.
    Raises FileNotFoundError with a helpful message if config is missing.

    Usage:
        config = load_config()
        season = config["nba"]["season"]
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at '{config_path}'. "
            "Make sure you're running the app from the project root directory."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_stat_color(value: float, avg: float) -> str:
    """
    Returns a color hex code based on whether a value is above or below average.
    Used to color-code stats in the dashboard.

    Args:
        value: The stat value to evaluate
        avg:   The player's average for that stat

    Returns:
        Green hex if above average, red if below, gray if near average
    """
    if value > avg * 1.1:
        return "#28a745"    # Green — above average
    elif value < avg * 0.9:
        return "#dc3545"    # Red — below average
    else:
        return "#6c757d"    # Gray — near average
