"""
services/ai_agent.py
====================
OpenAI integration that generates natural-language stat insights.

This module converts structured recent-game data + an ML prediction into a
short, readable analyst-style insight for the Dash UI. The prompt now also
includes two contextual signals used by the Random Forest predictor:
rest days before the upcoming game, and the opponent's defensive rating.
"""

import os
import time
from typing import Iterable, List

from dotenv import load_dotenv
from openai import OpenAI

from models.player import GameLog
from models.prediction import Prediction


load_dotenv()


# NBA league-average defensive rating — used as a sensible default whenever
# the caller does not have a real opponent rating available.
_LEAGUE_AVG_DEF_RATING: float = 112.0

# Threshold (in absolute stat units) under which a predicted value is
# treated as "near" the recent average rather than meaningfully up or
# down. 0.5 is small enough that a true bump still reads as a bump but
# noise-level wiggles don't generate a story.
_FLAT_DELTA_THRESHOLD: float = 0.5

# DEF_RATING bands used for matchup framing. Lower = tougher defense.
# Anything below 110 is treated as a stout defense; above 115 is treated
# as a softer defense. The middle band is "average".
_TOUGH_DEFENSE_CUTOFF: float = 110.0
_WEAK_DEFENSE_CUTOFF: float = 115.0


def _interpretive_stat_clause(
    stat: str,
    predicted: float,
    recent: float,
    opponent_def_rating: float,
    rest_days: int,
    is_playoff_game: bool,
) -> str:
    """Return one analyst-style sentence explaining WHY a stat moved.

    The clause is interpretive rather than mechanical: instead of "PTS is
    below the recent average", it says things like "PTS dips because the
    opponent's defense (DEF_RATING X) tends to deny clean looks for
    scorers like him". It picks the most likely *reason* for the move
    based on the available situational signals — defense quality, rest
    days, and playoff context — and falls back to recent form when no
    situational driver is dominant.

    The function is deliberately stat-aware: scoring narratives focus on
    clean looks, assist narratives focus on passing windows / pace, and
    rebound narratives focus on missed shots and minutes.
    """

    diff = predicted - recent
    # "Flat" gets its own short clause because writing a full why-sentence
    # for a 0.2-stat wiggle reads as overfitting.
    if abs(diff) <= _FLAT_DELTA_THRESHOLD:
        return f"{stat} holds steady around {predicted:.1f}, in line with his recent form."

    direction_up = diff > 0
    tough_defense = opponent_def_rating < _TOUGH_DEFENSE_CUTOFF
    weak_defense = opponent_def_rating > _WEAK_DEFENSE_CUTOFF
    extra_rest = rest_days >= 3
    no_rest = rest_days == 0

    # Each stat gets its own narrative menu so the language actually
    # reflects how that stat is generated on the floor (rim attacks,
    # passing reads, second-chance boards, etc).
    if stat == "PTS":
        if direction_up:
            if weak_defense:
                return (
                    f"PTS climbs to {predicted:.1f} because the model likes how he attacks "
                    f"softer defenses like this one (DEF_RATING {opponent_def_rating:.1f}) — "
                    f"more clean looks at the rim and from three."
                )
            if extra_rest:
                return (
                    f"PTS climbs to {predicted:.1f} because the {rest_days} days of rest "
                    f"typically gives him fresher legs and better shot quality late in games."
                )
            if is_playoff_game:
                return (
                    f"PTS climbs to {predicted:.1f} because playoff minutes are extended and "
                    f"he's getting more touches as a primary scoring option."
                )
            return (
                f"PTS climbs to {predicted:.1f} because his recent scoring trend is heading "
                f"the right way and the model expects it to carry."
            )
        # direction_down
        if tough_defense:
            return (
                f"PTS dips to {predicted:.1f} because this opponent (DEF_RATING "
                f"{opponent_def_rating:.1f}) is the kind of stout defense that has "
                f"historically denied his easy looks."
            )
        if no_rest:
            return (
                f"PTS dips to {predicted:.1f} because back-to-back fatigue typically drags "
                f"his scoring efficiency, especially on jumpers."
            )
        if is_playoff_game:
            return (
                f"PTS dips to {predicted:.1f} because playoff defenses key in on the lead "
                f"scorer and tighten their rotations on him."
            )
        return (
            f"PTS dips to {predicted:.1f} because his recent scoring has cooled and the "
            f"model is fading him slightly."
        )

    if stat == "AST":
        if direction_up:
            if weak_defense:
                return (
                    f"AST rises to {predicted:.1f} because softer defenses (DEF_RATING "
                    f"{opponent_def_rating:.1f}) tend to give up more drive-and-kick "
                    f"reads — exactly the looks he creates."
                )
            if extra_rest:
                return (
                    f"AST rises to {predicted:.1f} because the extra rest tends to sharpen "
                    f"his decision-making and reads in pick-and-roll."
                )
            if is_playoff_game:
                return (
                    f"AST rises to {predicted:.1f} because expanded playoff minutes mean "
                    f"more possessions where he's running the offense."
                )
            return (
                f"AST rises to {predicted:.1f} because his recent playmaking trend has been "
                f"strong and the model expects it to continue."
            )
        if tough_defense:
            return (
                f"AST dips to {predicted:.1f} because this defense (DEF_RATING "
                f"{opponent_def_rating:.1f}) shrinks passing windows with quick rotations "
                f"and physical on-ball pressure."
            )
        if no_rest:
            return (
                f"AST dips to {predicted:.1f} because back-to-back fatigue typically blunts "
                f"his playmaking sharpness."
            )
        if is_playoff_game:
            return (
                f"AST dips to {predicted:.1f} because playoff possessions slow down and "
                f"defenses are dialed in on his passing reads."
            )
        return (
            f"AST dips to {predicted:.1f} because his recent assist totals have cooled."
        )

    if stat == "REB":
        if direction_up:
            if is_playoff_game:
                return (
                    f"REB ticks up to {predicted:.1f} because playoff games tend to run "
                    f"longer with more contested possessions, creating extra board chances."
                )
            if weak_defense:
                return (
                    f"REB ticks up to {predicted:.1f} because softer defenses (DEF_RATING "
                    f"{opponent_def_rating:.1f}) tend to give up more long misses, which "
                    f"opens up second-chance opportunities."
                )
            if extra_rest:
                return (
                    f"REB ticks up to {predicted:.1f} because fresh legs help him hold "
                    f"position in box-out battles."
                )
            return (
                f"REB ticks up to {predicted:.1f} because his recent rebounding has been "
                f"trending up."
            )
        if tough_defense:
            return (
                f"REB dips to {predicted:.1f} because tougher defenses (DEF_RATING "
                f"{opponent_def_rating:.1f}) force fewer long misses, which usually means "
                f"fewer board chances."
            )
        if no_rest:
            return (
                f"REB dips to {predicted:.1f} because back-to-back fatigue cuts into "
                f"hustle plays and second-jumps."
            )
        return (
            f"REB dips to {predicted:.1f} because his recent rebounding has cooled."
        )

    # Unknown stat (shouldn't happen with PTS/AST/REB, but safe default).
    return f"{stat} projects at {predicted:.1f}, near his recent baseline."


def retry(max_attempts: int = 3, delay: float = 2.0):
    """
    Decorator factory that retries a function up to `max_attempts` times.

    The returned decorator wraps the target function in a loop: it will attempt
    to call the function, and on any exception it will sleep for `delay` seconds
    and retry until attempts are exhausted.

    Args:
        max_attempts: Maximum number of attempts before failing.
        delay: Seconds to sleep between failures.

    Returns:
        A decorator that applies retry behavior.

    Raises:
        RuntimeError: Raised by the wrapper on final failure with the last error.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise RuntimeError(f"Failed after {max_attempts} attempts. Last error: {last_error}") from last_error

        return wrapper

    return decorator


class AIAgent:
    """
    OpenAI-powered agent that generates concise, numbers-driven NBA insights.
    """

    def __init__(self, model: str = "gpt-3.5-turbo", max_tokens: int = 150) -> None:
        """
        Initialize the OpenAI client and configure model parameters.

        The API key is loaded from the environment via `os.environ.get`.

        Args:
            model: OpenAI chat model name.
            max_tokens: Maximum tokens for the generated insight.
        """

        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @retry(max_attempts=3, delay=2.0)
    def generate_insight(
        self,
        game_log: GameLog,
        prediction: Prediction,
        rest_days: int = 1,
        opponent_def_rating: float = _LEAGUE_AVG_DEF_RATING,
    ) -> str:
        """
        Generate an OpenAI insight about a player's upcoming performance.

        This method is decorated with retry logic to handle transient API failures.

        Args:
            game_log: Recent game history used as supporting evidence.
            prediction: ML model output (predicted value).
            rest_days: Days of rest before the upcoming game (0 = back-to-back).
            opponent_def_rating: Opponent defensive rating (lower = tougher).

        Returns:
            A concise 2–3 sentence insight string from the model.
        """

        # Build the full prompt including the two new contextual factors.
        prompt = self._build_prompt(game_log, prediction, rest_days, opponent_def_rating)

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an NBA sports analyst. Write a 2-3 sentence "
                        "outlook on the player's upcoming performance.\n\n"
                        "Explain WHY the predicted stat is above, below, or "
                        "near his recent average. Tie the explanation to one "
                        "of the situational signals provided: how he tends to "
                        "play against this style of defense (use the "
                        "DEF_RATING — lower is tougher), whether the rest "
                        "days help or hurt, or how playoff intensity / "
                        "minutes change his role.\n\n"
                        "Do NOT just say 'X is below the recent average'. "
                        "INTERPRET it. Use the numbers provided. Do not "
                        "invent stats."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content.strip()

    def generate_fallback_insight(
        self,
        game_log: GameLog,
        prediction: Prediction,
        rest_days: int = 1,
        opponent_def_rating: float = _LEAGUE_AVG_DEF_RATING,
    ) -> str:
        """
        Generate a deterministic insight without calling the OpenAI API.

        The fallback mentions rest days and opponent defense so the user still
        sees why the model produced the number it did — even offline.

        Args:
            game_log: Recent game history used for averages.
            prediction: ML model output (predicted value).
            rest_days: Days of rest before the upcoming game.
            opponent_def_rating: Opponent defensive rating.

        Returns:
            A manually formatted insight string.
        """

        avg_pts = game_log.average_points
        avg_ast = game_log.average_assists
        avg_reb = game_log.average_rebounds

        # Human-friendly description of rest context.
        if rest_days == 0:
            rest_desc = "no rest (back-to-back)"
        elif rest_days <= 2:
            rest_desc = f"{rest_days} day(s) of rest (typical)"
        else:
            rest_desc = f"{rest_days} days of rest (extra rest)"

        # Human-friendly description of opponent defense.
        if opponent_def_rating < 110.0:
            defense_desc = "a tough defense"
        elif opponent_def_rating > 115.0:
            defense_desc = "a weaker defense"
        else:
            defense_desc = "an average defense"

        # Single interpretive sentence that frames the projection in terms
        # of matchup style / rest / playoff context rather than a dry
        # comparison to the recent average.
        recent_for_stat = {
            "PTS": avg_pts,
            "AST": avg_ast,
            "REB": avg_reb,
        }.get(prediction.stat_category, 0.0)
        why_sentence = _interpretive_stat_clause(
            stat=prediction.stat_category,
            predicted=prediction.predicted_value,
            recent=recent_for_stat,
            opponent_def_rating=opponent_def_rating,
            rest_days=rest_days,
            is_playoff_game=False,
        )

        return (
            f"{game_log.player_name} has averaged {avg_pts:.1f} PTS, {avg_ast:.1f} AST, "
            f"and {avg_reb:.1f} REB over the last {game_log.game_count()} games, with "
            f"{rest_desc} and facing {defense_desc} "
            f"(DEF_RATING {opponent_def_rating:.1f}). "
            f"{why_sentence}"
        )

    @retry(max_attempts=3, delay=2.0)
    def generate_insight_multi(
        self,
        game_log: GameLog,
        predictions: Iterable[Prediction],
        rest_days: int = 1,
        opponent_def_rating: float = _LEAGUE_AVG_DEF_RATING,
        is_playoff_game: bool = False,
        playoff_series_label: str = "",
    ) -> str:
        """
        Generate a single OpenAI insight covering multiple stat predictions at once.

        Sending one combined prompt is cheaper than three separate API calls and
        yields a more cohesive, analyst-style paragraph than gluing together
        three independent sentences.

        Args:
            game_log: Recent game history used as supporting evidence.
            predictions: Iterable of `Prediction` objects (typically one per
                stat — PTS / AST / REB).
            rest_days: Days of rest before the upcoming game (0 = back-to-back).
            opponent_def_rating: Opponent defensive rating (lower = tougher).
            is_playoff_game: True when the upcoming game is a postseason
                game; the prompt explicitly tells the model so the insight
                can reference series stakes / matchup history.
            playoff_series_label: Optional human-readable round/series text
                (e.g. "Round 1, Game 3 — BOS leads 2-0").

        Returns:
            A 2–4 sentence insight string from the model.
        """

        preds_list: List[Prediction] = list(predictions)
        prompt = self._build_prompt_multi(
            game_log,
            preds_list,
            rest_days,
            opponent_def_rating,
            is_playoff_game=is_playoff_game,
            playoff_series_label=playoff_series_label,
        )

        response = self._client.chat.completions.create(
            model=self.model,
            # Bump max tokens since we're now writing per-stat narratives.
            max_tokens=max(self.max_tokens, 280),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an NBA sports analyst. Write a 3-5 sentence "
                        "outlook for a player's upcoming game.\n\n"
                        "For EACH predicted stat (PTS, AST, REB), explain WHY "
                        "it is above, below, or near the player's recent "
                        "average. Tie the explanation to one of the provided "
                        "situational signals: how he tends to play against "
                        "this style of defense (use the DEF_RATING — lower is "
                        "tougher), whether the rest days help or hurt, or how "
                        "playoff minutes / intensity change his role.\n\n"
                        "Do NOT just restate the comparison ('PTS is below the "
                        "recent average'). INTERPRET it. Good examples:\n"
                        "  - 'PTS dips because this defense is built around "
                        "denying drives, which has historically limited his "
                        "easy looks.'\n"
                        "  - 'AST climbs because softer defenses give up more "
                        "drive-and-kick reads, exactly the looks he creates.'\n"
                        "  - 'REB ticks up because playoff games run longer "
                        "and produce more contested possessions.'\n\n"
                        "Use the numbers provided. Do not invent stats."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content.strip()

    def generate_fallback_insight_multi(
        self,
        game_log: GameLog,
        predictions: Iterable[Prediction],
        rest_days: int = 1,
        opponent_def_rating: float = _LEAGUE_AVG_DEF_RATING,
        is_playoff_game: bool = False,
        playoff_series_label: str = "",
    ) -> str:
        """
        Deterministic offline insight covering all provided predictions.

        Mirrors `generate_fallback_insight` but mentions every stat in one
        paragraph so the predictor UI still has something to display when the
        OpenAI API is unavailable.

        Args:
            game_log: Recent game history used for averages.
            predictions: Iterable of `Prediction` objects (one per stat).
            rest_days: Days of rest before the upcoming game.
            opponent_def_rating: Opponent defensive rating.

        Returns:
            A manually formatted insight string.
        """

        preds_list: List[Prediction] = list(predictions)
        avg_pts = game_log.average_points
        avg_ast = game_log.average_assists
        avg_reb = game_log.average_rebounds

        # Short verbal summary of each predicted stat.
        pred_chunks = ", ".join(
            f"{p.predicted_value:.1f} {p.stat_category}" for p in preds_list
        ) or "no predictions available"

        # Human-friendly rest context.
        if rest_days == 0:
            rest_desc = "no rest (back-to-back)"
        elif rest_days <= 2:
            rest_desc = f"{rest_days} day(s) of rest (typical)"
        else:
            rest_desc = f"{rest_days} days of rest (extra rest)"

        # Human-friendly defense context.
        if opponent_def_rating < 110.0:
            defense_desc = "a tough defense"
        elif opponent_def_rating > 115.0:
            defense_desc = "a weaker defense"
        else:
            defense_desc = "an average defense"

        # Surface playoff context so the offline insight makes sense too.
        playoff_recent = sum(
            1 for g in game_log.games if getattr(g, "is_playoff_game", False)
        )
        recent_window_desc = f"the last {game_log.game_count()} games"
        if playoff_recent:
            recent_window_desc += f" ({playoff_recent} from the playoffs)"
        playoff_clause = ""
        if is_playoff_game:
            playoff_clause = (
                f" The upcoming game is a playoff game"
                + (f" ({playoff_series_label})" if playoff_series_label else "")
                + "."
            )

        # Per-stat interpretive sentences. Each one names the situational
        # driver (matchup style, rest, playoff intensity) instead of just
        # restating "above/below the recent average".
        recent_avg_by_stat = {
            "PTS": avg_pts,
            "AST": avg_ast,
            "REB": avg_reb,
        }
        why_sentences: List[str] = []
        for p in preds_list:
            recent = recent_avg_by_stat.get(p.stat_category)
            if recent is None:
                continue
            why_sentences.append(
                _interpretive_stat_clause(
                    stat=p.stat_category,
                    predicted=p.predicted_value,
                    recent=recent,
                    opponent_def_rating=opponent_def_rating,
                    rest_days=rest_days,
                    is_playoff_game=is_playoff_game,
                )
            )
        why_block = " ".join(why_sentences) if why_sentences else ""

        return (
            f"{game_log.player_name} has averaged {avg_pts:.1f} PTS, {avg_ast:.1f} AST, "
            f"and {avg_reb:.1f} REB over {recent_window_desc}, with {rest_desc} and "
            f"facing {defense_desc} (DEF_RATING {opponent_def_rating:.1f}). "
            f"{why_block}"
            f"{playoff_clause}"
        )

    def _build_prompt_multi(
        self,
        game_log: GameLog,
        predictions: List[Prediction],
        rest_days: int,
        opponent_def_rating: float,
        is_playoff_game: bool = False,
        playoff_series_label: str = "",
    ) -> str:
        """
        Build the user prompt sent to OpenAI for a multi-stat insight.

        Includes the same contextual block as the single-stat prompt but
        lists every prediction together so the model can reason about them
        jointly (e.g. "scoring dips but playmaking spikes vs. tough D").

        Args:
            game_log: Recent game history.
            predictions: List of `Prediction` objects (one per stat).
            rest_days: Days of rest before the upcoming game.
            opponent_def_rating: Opponent defensive rating.

        Returns:
            A prompt string for the chat completion endpoint.
        """

        # game_log.games is newest-first; first 5 are the last 5 games played.
        recent_games = game_log.games[:5]
        last5_pts = ", ".join(f"{g.points:.1f}" for g in recent_games) if recent_games else "N/A"
        last5_ast = ", ".join(f"{g.assists:.1f}" for g in recent_games) if recent_games else "N/A"
        last5_reb = ", ".join(f"{g.rebounds:.1f}" for g in recent_games) if recent_games else "N/A"

        # Bullet-list each prediction so the model sees them as structured data.
        pred_lines = "\n".join(
            f"  - {p.stat_category}: {p.predicted_value:.1f}" for p in predictions
        ) or "  - (none)"

        # Tag postseason rows so the model knows which of the recent games
        # came from the playoffs (where intensity / rotations are different).
        playoff_recent = sum(
            1 for g in game_log.games if getattr(g, "is_playoff_game", False)
        )
        recent_window_line = (
            f"Recent games window: last {game_log.game_count()} played "
            f"({playoff_recent} playoff, {game_log.game_count() - playoff_recent} regular season)\n"
        )

        playoff_line = ""
        if is_playoff_game:
            playoff_line = (
                f"Postseason context: this is a PLAYOFF game"
                + (f" — {playoff_series_label}" if playoff_series_label else "")
                + ". Defensive intensity is typically higher and rotations tighten.\n"
            )

        return (
            f"Player: {game_log.player_name}\n"
            f"Last 5 games (PTS): {last5_pts}\n"
            f"Last 5 games (AST): {last5_ast}\n"
            f"Last 5 games (REB): {last5_reb}\n"
            f"Recent averages: {game_log.average_points:.1f} PTS, "
            f"{game_log.average_assists:.1f} AST, {game_log.average_rebounds:.1f} REB\n"
            f"{recent_window_line}"
            f"Predictions for next game:\n{pred_lines}\n"
            f"Rest days before next game: {rest_days}\n"
            f"Opponent defensive rating: {opponent_def_rating:.1f} "
            f"(NBA average is ~112. Lower = tougher defense, higher = weaker defense)\n"
            f"{playoff_line}"
            f"Model used: Random Forest (non-linear, accounts for rest and defense)\n"
        )

    def _build_prompt(
        self,
        game_log: GameLog,
        prediction: Prediction,
        rest_days: int = 1,
        opponent_def_rating: float = _LEAGUE_AVG_DEF_RATING,
    ) -> str:
        """
        Build the user prompt sent to OpenAI for generating an insight.

        The prompt includes:
        - Player name
        - Last 5 game values for PTS/AST/REB
        - Season averages (approximated using the provided log averages)
        - ML prediction (predicted value only)
        - Rest days before the upcoming game
        - Opponent defensive rating (with a note on how to interpret it)
        - The model family used (Random Forest)

        Args:
            game_log: Recent game history.
            prediction: ML model output.
            rest_days: Days of rest before the upcoming game.
            opponent_def_rating: Opponent defensive rating.

        Returns:
            A prompt string to send to the OpenAI chat completion endpoint.
        """

        # game_log.games is newest-first; first 5 are the last 5 games played.
        recent_games = game_log.games[:5]
        last5_pts = ", ".join(f"{g.points:.1f}" for g in recent_games) if recent_games else "N/A"
        last5_ast = ", ".join(f"{g.assists:.1f}" for g in recent_games) if recent_games else "N/A"
        last5_reb = ", ".join(f"{g.rebounds:.1f}" for g in recent_games) if recent_games else "N/A"

        return (
            f"Player: {game_log.player_name}\n"
            f"Last 5 games (PTS): {last5_pts}\n"
            f"Last 5 games (AST): {last5_ast}\n"
            f"Last 5 games (REB): {last5_reb}\n"
            f"Season averages (from provided data): "
            f"{game_log.average_points:.1f} PTS, {game_log.average_assists:.1f} AST, "
            f"{game_log.average_rebounds:.1f} REB\n"
            f"ML prediction: {prediction.predicted_value:.1f} {prediction.stat_category}\n"
            f"Rest days before next game: {rest_days}\n"
            f"Opponent defensive rating: {opponent_def_rating:.1f} "
            f"(NBA average is ~112. Lower = tougher defense, higher = weaker defense)\n"
            f"Model used: Random Forest (non-linear, accounts for rest and defense)\n"
        )
