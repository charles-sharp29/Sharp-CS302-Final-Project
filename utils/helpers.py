"""
utils/helpers.py
================
Shared utility functions used across the NBA analytics Streamlit app.

These helpers centralize common tasks like safe type conversion, formatting for
UI display, loading configuration, and choosing conditional colors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convert an arbitrary value to float safely.

    Args:
        value: Any input value (string, number, None, etc.).
        default: Value returned when conversion fails.

    Returns:
        A float representation of value, or default on failure.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Convert an arbitrary value to int safely.

    Args:
        value: Any input value (string, number, None, etc.).
        default: Value returned when conversion fails.

    Returns:
        An int representation of value, or default on failure.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def format_stat(value: float, decimal_places: int = 1) -> str:
    """
    Format a float to N decimal places as a string.

    Args:
        value: Numeric value to format.
        decimal_places: Number of digits after the decimal point.

    Returns:
        The formatted number as a string.
    """

    return f"{value:.{decimal_places}f}"


def format_percentage(value: float) -> str:
    """
    Convert a decimal proportion to a percentage string.

    Example:
        0.523 -> "52.3%"

    Args:
        value: Decimal proportion (e.g., 0.523 for 52.3%).

    Returns:
        Percentage string with one decimal place.
    """

    return f"{value * 100.0:.1f}%"


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load a YAML configuration file and return it as a dictionary.

    Args:
        config_path: Path to the YAML config file (defaults to "config.yaml").

    Returns:
        A dictionary of configuration values loaded from YAML.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'. "
            "Create it or run the app from the project root so relative paths resolve."
        )

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if isinstance(data, dict) else {}


def get_stat_color(value: float, avg: float) -> str:
    """
    Choose a Bootstrap-like hex color based on performance vs. average.

    Rules:
        - Green  : value > avg * 1.1
        - Red    : value < avg * 0.9
        - Gray   : otherwise

    Args:
        value: The stat value being evaluated.
        avg: The baseline average for comparison.

    Returns:
        A hex color string ("#28a745", "#dc3545", or "#6c757d").
    """

    if value > avg * 1.1:
        return "#28a745"
    if value < avg * 0.9:
        return "#dc3545"
    return "#6c757d"

