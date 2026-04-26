"""
services/nba_fetcher.py
=======================
Fetches NBA player data using the `nba_api` package.

This module is the service-layer boundary between the Dash app and external
NBA data sources. It returns validated model objects (`Player`, `PlayerStats`,
`GameLog`) rather than raw dictionaries/DataFrames, and also exposes helpers
for two contextual signals used by the Random Forest predictor:

- `get_rest_days`            : days since the player's most recent game
- `get_opponent_def_rating`  : opponent team defensive rating
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
from nba_api.stats.endpoints import (
    commonplayerinfo,
    leaguedashteamstats,
    playercareerstats,
    playergamelog,
    scheduleleaguev2,
)
from nba_api.stats.static import players, teams

from models.player import GameLog, Player, PlayerStats
from utils.helpers import safe_float, safe_int


def _build_team_id_to_abbr_map() -> Dict[int, str]:
    """
    Build a lookup from NBA team id to tricode, e.g. 1610612738 -> "BOS".

    LeagueDashTeamStats with measure_type_detailed_defense="Defense" returns
    TEAM_ID + TEAM_NAME but NOT TEAM_ABBREVIATION, so we need this map to
    convert rows back into tricodes the rest of the app speaks in.
    """

    try:
        return {
            int(t["id"]): str(t["abbreviation"]).upper()
            for t in teams.get_teams()
            if t.get("id") is not None and t.get("abbreviation")
        }
    except Exception:
        return {}


# Build once at import time — the static teams list is tiny and stable.
_TEAM_ID_TO_ABBR: Dict[int, str] = _build_team_id_to_abbr_map()


# NBA league-average defensive rating used as a fallback whenever we cannot
# resolve a real value (unknown team, API failure, missing column, etc.).
_LEAGUE_AVG_DEF_RATING: float = 112.0

# Rest-day bounds. Anything over two weeks is treated as "fully rested" and
# clamped so extreme off-season gaps do not leak into the feature matrix.
_MIN_REST_DAYS: int = 0
_MAX_REST_DAYS: int = 14

# NBA game-log dates are reported in Eastern time (e.g. "APR 12, 2026"), and
# schedule tip-off times are UTC. Converting UTC to ET before taking the
# calendar date keeps rest-day math correct — a 00:30 UTC tip-off is actually
# the previous evening in ET, so naive UTC date math would overcount by a day.
_NBA_TZ: ZoneInfo = ZoneInfo("America/New_York")

# Season types we pull when assembling a player's "recent games". During the
# postseason the most recent games are playoff games, so we always fetch both
# and merge — sorting by GAME_DATE puts the latest games (whichever type) at
# the top. When fetched in the off-season / regular season, the playoff
# query simply returns an empty frame and the merge is a no-op.
_SEASON_TYPES_FOR_GAMELOG: Tuple[str, ...] = ("Regular Season", "Playoffs")


class NBAFetcher:
    """
    Service object that wraps `nba_api` calls and returns app model objects.

    The class is designed to be used as a context manager so callers can use a
    clean `with` block and have a consistent lifecycle.
    """

    def __init__(self, season: str = "2024-25", recent_games: int = 10) -> None:
        """
        Create a new NBAFetcher.

        Args:
            season: NBA season string used by nba_api (e.g., "2024-25").
            recent_games: Number of most-recent games to return in game logs.
        """

        self.season = season
        self.recent_games = recent_games
        self._request_delay = 0.6

    def __enter__(self) -> "NBAFetcher":
        """
        Enter a context manager block.

        Returns:
            The fetcher instance itself.
        """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit a context manager block.

        This does not suppress exceptions.

        Returns:
            False to allow any exception to propagate.
        """

        return False

    def search_player(self, name: str) -> Optional[Player]:
        """
        Search for a player by full name and return a populated `Player`.

        This uses `players.find_players_by_full_name()` to locate candidate
        matches. If a match is found, it fetches additional details from
        `CommonPlayerInfo` (team and position).

        Args:
            name: Player name (full or partial) to search for.

        Returns:
            A `Player` if found; otherwise None.

        Raises:
            RuntimeError: If the nba_api request fails unexpectedly.
        """

        try:
            results = players.find_players_by_full_name(name)
            if not results:
                return None

            match = results[0]
            player_id = safe_int(match.get("id"))
            full_name = str(match.get("full_name", name))

            time.sleep(self._request_delay)
            info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            info_df = info.get_data_frames()[0]

            team = "Unknown"
            position = "Unknown"
            team_abbr: Optional[str] = None
            if not info_df.empty:
                team = str(info_df.get("TEAM_NAME", ["Unknown"])[0])
                position = str(info_df.get("POSITION", ["Unknown"])[0])
                # Also pull the tricode (e.g. "LAL") so downstream calls do not
                # have to re-query CommonPlayerInfo just to look up the team abbr.
                if "TEAM_ABBREVIATION" in info_df.columns:
                    raw_abbr = info_df["TEAM_ABBREVIATION"].iloc[0]
                    if raw_abbr and not pd.isna(raw_abbr):
                        team_abbr = str(raw_abbr).strip().upper()

            return Player(
                name=full_name,
                team=team,
                position=position,
                player_id=player_id,
                team_abbreviation=team_abbr,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to search/fetch player info for '{name}'. Error: {e}") from e

    def get_game_log(self, player_id: int) -> GameLog:
        """
        Fetch a player's recent games (regular season + playoffs combined).

        We fetch the regular-season log and, if a postseason is in progress,
        the playoff log too, then sort the combined frame by GAME_DATE and
        return the configured number of most-recent games. This is critical
        during the playoffs: by default `PlayerGameLog` only returns
        regular-season rows, so the dashboard would show "the last 10
        regular-season games" and miss the playoff context the user
        actually cares about.

        Args:
            player_id: NBA API internal player identifier.

        Returns:
            A `GameLog` containing up to `self.recent_games` most recent
            games across both Regular Season and Playoffs.

        Raises:
            RuntimeError: If both nba_api requests fail or no data is
                returned for either season type.
        """

        try:
            df = self._fetch_combined_player_log(player_id)
            if df.empty:
                raise RuntimeError(f"No game log data returned for player_id={player_id}.")

            df = df.head(self.recent_games)

            player_name = str(df["PLAYER_NAME"].iloc[0]) if "PLAYER_NAME" in df.columns else str(player_id)
            games = list(self._parse_game_rows(df))
            return GameLog(player_name=player_name, games=games)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to fetch game log for player_id={player_id}. Error: {e}") from e

    def _fetch_combined_player_log(self, player_id: int) -> pd.DataFrame:
        """Fetch and merge regular-season + playoff game logs.

        Each season-type query is independent: a failure on the playoffs
        endpoint (e.g. team didn't make the postseason) must not poison the
        regular-season data we already have. Rows are tagged with a
        SEASON_TYPE column so downstream parsers can surface the postseason
        flag. The frame is sorted by GAME_DATE descending so the
        most-recent games (whichever type) bubble to the top.
        """

        frames: list[pd.DataFrame] = []
        for season_type in _SEASON_TYPES_FOR_GAMELOG:
            try:
                time.sleep(self._request_delay)
                endpoint = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=self.season,
                    season_type_all_star=season_type,
                )
                part = endpoint.get_data_frames()[0]
                if part is None or part.empty:
                    continue
                # Tag rows so _parse_game_rows can flag playoff games for the UI.
                part = part.copy()
                part["SEASON_TYPE"] = season_type
                frames.append(part)
            except Exception:
                # One season type failing is fine — fall back to whatever we
                # have. Both failing is handled by the empty-check below.
                continue

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        # GAME_DATE arrives as "APR 10, 2026"; coerce to a real datetime for
        # a stable sort, then drop the helper column.
        if "GAME_DATE" in combined.columns:
            combined["_GAME_DATE_DT"] = pd.to_datetime(
                combined["GAME_DATE"], format="%b %d, %Y", errors="coerce"
            )
            combined = combined.sort_values(
                "_GAME_DATE_DT", ascending=False, na_position="last"
            ).drop(columns=["_GAME_DATE_DT"])
        return combined.reset_index(drop=True)

    def get_season_avg(self, player_id: int, player_name: str) -> PlayerStats:
        """
        Fetch a player's season average stats for the configured season.

        This uses career totals by season and converts totals into per-game
        averages using GP. If the configured season isn't present, the most
        recent available season is used.

        Args:
            player_id: NBA API internal player identifier.
            player_name: Player name to attach to the returned PlayerStats.

        Returns:
            A `PlayerStats` object with `is_season_average=True`.

        Raises:
            RuntimeError: If the nba_api request fails or parsing fails.
        """

        try:
            time.sleep(self._request_delay)
            endpoint = playercareerstats.PlayerCareerStats(player_id=player_id)
            df = endpoint.get_data_frames()[0]

            if df.empty:
                raise RuntimeError(f"No career stats data returned for player_id={player_id}.")

            season_df = df[df["SEASON_ID"] == self.season] if "SEASON_ID" in df.columns else pd.DataFrame()
            if season_df.empty:
                season_df = df.tail(1)

            row = season_df.iloc[0]
            gp = safe_int(row.get("GP", 0))
            denom = max(gp, 1)

            return PlayerStats(
                player_name=player_name,
                points=safe_float(row.get("PTS", 0.0)) / denom,
                assists=safe_float(row.get("AST", 0.0)) / denom,
                rebounds=safe_float(row.get("REB", 0.0)) / denom,
                fg_percentage=safe_float(row.get("FG_PCT", 0.0)),
                fg3_percentage=safe_float(row.get("FG3_PCT", 0.0)),
                games_played=gp,
                is_season_average=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch season averages for player_id={player_id}. Error: {e}") from e

    def get_rest_days(self, player_id: int, next_game_dt: datetime) -> int:
        """
        Compute how many days of rest a player has before their next scheduled game.

        The returned value is the calendar gap between the player's most recent
        played game and the team's NEXT scheduled game, adjusted so it lines up
        with how NBA broadcasts count rest:

            gap = (next_game_date - last_played_date).days

            - gap == 1 (back-to-back)               -> 0 rest days
            - gap == 2 (one day off in between)     -> 1 rest day
            - gap >= 3 (extended break)             -> gap - 2 rest days,
              i.e. one day is treated as a travel / shoot-around day and does
              not count as true rest. So Apr 12 -> Apr 18 (gap 6) = 4.

        Working off the league schedule (not `datetime.now()`) means this still
        returns sensible values even after the regular season ends.

        Args:
            player_id: NBA API internal player identifier.
            next_game_dt: Datetime of the upcoming game (from `get_next_game`).

        Returns:
            An integer in the range [0, 14]. Defaults to 1 on any failure.
        """

        try:
            # Use the combined regular-season + playoffs log so the "most
            # recent game" we measure rest from is whichever happened
            # latest, not just the last regular-season game.
            df = self._fetch_combined_player_log(player_id)

            # Defensive check: empty frame or missing date column means we cannot compute.
            if df.empty or "GAME_DATE" not in df.columns:
                return 1

            # The top row is the most recent game across both season types.
            latest_game_date_str = str(df["GAME_DATE"].iloc[0])
            # nba_api returns dates like "APR 10, 2025"; parse that format.
            latest_dt = datetime.strptime(latest_game_date_str, "%b %d, %Y")

            # Convert the UTC tip-off to Eastern time before taking its
            # calendar date. NBA game-log dates are ET ("APR 12"), and a late
            # ET tip-off can roll over past midnight UTC, so a plain .date()
            # on the UTC datetime would overcount rest by 1 day.
            if next_game_dt.tzinfo is None:
                # If somehow naive, assume the caller already did the math.
                next_date = next_game_dt.date()
            else:
                next_date = next_game_dt.astimezone(_NBA_TZ).date()
            latest_date = latest_dt.date()

            gap = (next_date - latest_date).days

            # Piecewise rule (see docstring):
            #   - gap <= 2 uses the textbook "days in between" definition so
            #     back-to-backs stay 0 and Mon->Wed stays 1.
            #   - gap >= 3 subtracts an extra day to absorb the travel /
            #     shoot-around day, so a 6-day gap reads as 4 rest days.
            if gap <= 2:
                raw_rest = gap - 1
            else:
                raw_rest = gap - 2

            # Clamp to [0, 14] so off-season gaps or stale schedules cannot
            # distort the predictor's feature matrix.
            return max(_MIN_REST_DAYS, min(_MAX_REST_DAYS, int(raw_rest)))
        except Exception:
            # Any failure (network, parsing, unexpected format) falls back to 1
            # which represents a typical "one day off" baseline.
            return 1

    def get_opponent_def_rating(self, team_abbreviation: str) -> float:
        """
        Return the opponent team's defensive rating (DEF_RATING) for the season.

        A higher DEF_RATING means a worse defense (points allowed per 100
        possessions go up, so it's easier to score). A lower value means a
        tougher defense.

        Args:
            team_abbreviation: 2–3 letter team code, e.g. "BOS", "GSW", "MIA".

        Returns:
            The team's DEF_RATING as a float. Falls back to 112.0 (league
            average) if the team is not found or the API call fails.
        """

        try:
            # Short-circuit on empty input before making an API call.
            if not team_abbreviation or not str(team_abbreviation).strip():
                return _LEAGUE_AVG_DEF_RATING

            # Delegate to the bulk helper and grab the requested team.
            # Single-team lookups still cost one API call but benefit from the
            # same TEAM_ID → abbreviation join used everywhere else.
            ratings = self.get_all_team_def_ratings()
            target = str(team_abbreviation).strip().upper()
            value = ratings.get(target)
            return float(value) if value is not None else _LEAGUE_AVG_DEF_RATING
        except Exception:
            # On any error (network, schema change, etc.) fall back to league average.
            return _LEAGUE_AVG_DEF_RATING

    def get_all_team_def_ratings(self) -> Dict[str, float]:
        """
        Return a map of every team tricode to its defensive rating.

        During the playoffs we prefer the postseason DEF_RATING because
        defensive intensity (and rotations) shift dramatically once the
        playoffs start — a team's regular-season number can be a poor
        predictor of how they're actually defending right now. We fetch
        playoffs first, then merge regular-season ratings underneath as a
        fallback for teams that didn't make the postseason.

        Returns:
            A dict like {"BOS": 109.4, "LAL": 114.1, ...}. Returns an empty
            dict on total API / parsing failure; callers should treat an
            empty result as "defense data unavailable" and surface an error
            to the user rather than silently substituting the league
            average.
        """

        # Regular season is always populated; playoffs is only populated
        # once the postseason starts. Layer playoffs on top so playoff
        # numbers win when they exist.
        regular = self._fetch_team_def_ratings("Regular Season")
        playoffs = self._fetch_team_def_ratings("Playoffs")

        merged: Dict[str, float] = {}
        merged.update(regular)
        merged.update(playoffs)
        return merged

    def _fetch_team_def_ratings(self, season_type: str) -> Dict[str, float]:
        """Fetch DEF_RATING for every team for a given season type.

        Returns an empty dict on any API / parsing failure or when the
        endpoint legitimately returns no rows (e.g. asking for "Playoffs"
        before the postseason starts).
        """

        try:
            time.sleep(self._request_delay)
            endpoint = leaguedashteamstats.LeagueDashTeamStats(
                season=self.season,
                # "Defense" exposes DEF_RATING (points allowed per 100 possessions).
                measure_type_detailed_defense="Defense",
                season_type_all_star=season_type,
            )
            df = endpoint.get_data_frames()[0]

            # The "Defense" measure type returns TEAM_ID + TEAM_NAME but NOT
            # TEAM_ABBREVIATION, so we join on TEAM_ID via the static team list.
            if df.empty or "TEAM_ID" not in df.columns or "DEF_RATING" not in df.columns:
                return {}

            ratings: Dict[str, float] = {}
            for _, row in df.iterrows():
                team_id_raw = row.get("TEAM_ID")
                rating_raw = row.get("DEF_RATING")
                if (
                    team_id_raw is None or pd.isna(team_id_raw)
                    or rating_raw is None or pd.isna(rating_raw)
                ):
                    continue
                try:
                    team_id = int(team_id_raw)
                except (TypeError, ValueError):
                    continue
                abbr = _TEAM_ID_TO_ABBR.get(team_id)
                if not abbr:
                    continue
                ratings[abbr] = safe_float(rating_raw)
            return ratings
        except Exception:
            return {}

    def get_next_game(self, team_abbreviation: str) -> Optional[Tuple[str, datetime]]:
        """
        Resolve the next scheduled game for a given team.

        Thin wrapper that returns just the opponent tricode and tip-off
        time. For playoff awareness use `get_next_game_detail`.
        """

        detail = self.get_next_game_detail(team_abbreviation)
        if detail is None:
            return None
        return detail[0], detail[1]

    def get_next_game_detail(
        self, team_abbreviation: str
    ) -> Optional[Tuple[str, datetime, bool, str]]:
        """
        Like `get_next_game` but also returns playoff context.

        Returns:
            A tuple of (opponent_tricode, game_datetime_utc,
            is_playoff_game, series_label). `series_label` is the
            human-readable round/series text from the league schedule
            (e.g. "Round 1, Game 3 — BOS leads series 2-0") and is the
            empty string for regular-season games.
        """

        try:
            if not team_abbreviation or not str(team_abbreviation).strip():
                return None

            target = str(team_abbreviation).strip().upper()

            schedule = self._fetch_league_schedule()
            if not schedule:
                return None

            now_utc = datetime.now(timezone.utc)

            next_opp: Optional[str] = None
            next_dt: Optional[datetime] = None
            next_is_playoff: bool = False
            next_series_label: str = ""

            game_dates = schedule.get("leagueSchedule", {}).get("gameDates", []) or []
            for game_date_entry in game_dates:
                for game in game_date_entry.get("games", []) or []:
                    dt_str = game.get("gameDateTimeUTC")
                    if not dt_str:
                        continue
                    try:
                        game_dt = datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))
                    except ValueError:
                        continue

                    if game_dt <= now_utc:
                        continue

                    home = str((game.get("homeTeam") or {}).get("teamTricode", "")).upper()
                    away = str((game.get("awayTeam") or {}).get("teamTricode", "")).upper()

                    if home == target:
                        opponent = away
                    elif away == target:
                        opponent = home
                    else:
                        continue

                    if next_dt is None or game_dt < next_dt:
                        next_dt = game_dt
                        next_opp = opponent
                        # Playoff detection: NBA.com's schedule schema marks
                        # postseason games with a non-empty `gameLabel`
                        # ("Round 1", "Conf. Semifinals", "NBA Finals", …)
                        # AND a `weekName` containing "Playoffs". We accept
                        # either signal so we stay robust if the league
                        # tweaks one of them.
                        game_label = str(game.get("gameLabel") or "").strip()
                        game_sub_label = str(game.get("gameSubLabel") or "").strip()
                        week_name = str(game.get("weekName") or "").strip()
                        next_is_playoff = bool(game_label) or "playoff" in week_name.lower()
                        # Build a friendly series label for the UI badge.
                        if next_is_playoff:
                            parts = [p for p in (game_label, game_sub_label) if p]
                            next_series_label = " — ".join(parts) if parts else (
                                week_name or "Playoffs"
                            )
                        else:
                            next_series_label = ""

            if next_opp and next_dt is not None:
                return next_opp, next_dt, next_is_playoff, next_series_label
            return None
        except Exception:
            # Any unexpected schema / network / parsing failure → no game.
            return None

    def get_next_opponent(self, team_abbreviation: str) -> Optional[str]:
        """
        Thin wrapper around `get_next_game` that returns just the opponent tricode.

        Kept for callers that don't need the tip-off datetime.

        Args:
            team_abbreviation: Team tricode of the player's current team.

        Returns:
            Opponent tricode, or None if no upcoming game is found.
        """

        result = self.get_next_game(team_abbreviation)
        return result[0] if result else None

    def _fetch_league_schedule(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the full NBA league schedule via nba_api's ScheduleLeagueV2.

        This is the same data that powers NBA.com's schedule pages: every
        regular-season and postseason game for the current league year, with
        UTC tipoff times and home/away team tricodes. We route through
        nba_api so we inherit its working request headers (the raw CDN URL
        returns HTTP 403 to non-browser clients).

        Returns:
            The parsed JSON as a dict with shape {"leagueSchedule": ...}, or
            None on any network / parsing failure.
        """

        try:
            # Polite delay to match every other nba_api call in this module.
            time.sleep(self._request_delay)
            endpoint = scheduleleaguev2.ScheduleLeagueV2()
            return endpoint.get_dict()
        except Exception:
            # Deliberately swallow: callers treat None as "schedule unavailable"
            # and fall back to sensible defaults (league-average DEF_RATING, etc.).
            return None

    def _parse_game_rows(self, df: pd.DataFrame) -> Generator[PlayerStats, None, None]:
        """
        Parse a game log DataFrame into `PlayerStats` objects.

        This is implemented as a generator to demonstrate `yield` and allow
        streaming conversion of rows to objects. Each yielded object carries
        the game date and opponent tricode parsed from the `MATCHUP` column,
        which the predictor uses to compute per-game rest days and look up
        the opponent's defensive rating.

        Args:
            df: DataFrame returned by `PlayerGameLog.get_data_frames()[0]`.

        Yields:
            One `PlayerStats` per row.
        """

        for _, row in df.iterrows():
            # MATCHUP format from nba_api is either "LAL vs. BOS" (home) or
            # "LAL @ BOS" (away). The opponent tricode is always the last token.
            raw_matchup = str(row.get("MATCHUP", "")).strip()
            opponent_abbr: Optional[str] = None
            if raw_matchup:
                tokens = raw_matchup.split()
                if tokens:
                    # Last token = opponent tricode; strip any stray punctuation.
                    opponent_abbr = tokens[-1].strip(".,").upper() or None

            # GAME_DATE comes in nba_api as "APR 10, 2025"; we keep it as the
            # raw string here and parse/diff it inside the predictor where
            # per-game rest days are computed.
            raw_date = row.get("GAME_DATE")
            game_date: Optional[str] = (
                str(raw_date).strip() if raw_date is not None and not pd.isna(raw_date) else None
            )

            # SEASON_TYPE is set by _fetch_combined_player_log when it merges
            # regular-season and playoff frames. Older callers that pass a
            # raw frame may not have it, so default to False.
            season_type_raw = row.get("SEASON_TYPE") if "SEASON_TYPE" in row.index else None
            is_playoff = (
                isinstance(season_type_raw, str)
                and season_type_raw.strip().lower() == "playoffs"
            )

            yield PlayerStats(
                player_name=str(row.get("PLAYER_NAME", "")),
                points=safe_float(row.get("PTS", 0.0)),
                assists=safe_float(row.get("AST", 0.0)),
                rebounds=safe_float(row.get("REB", 0.0)),
                fg_percentage=safe_float(row.get("FG_PCT", 0.0)),
                fg3_percentage=safe_float(row.get("FG3_PCT", 0.0)),
                games_played=1,
                is_season_average=False,
                game_date=game_date,
                opponent_abbreviation=opponent_abbr,
                is_playoff_game=is_playoff,
            )
