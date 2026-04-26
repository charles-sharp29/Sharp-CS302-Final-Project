"""
app.py
======
Dash entry point for the NBA analytics project.

Run with:
    python app.py
"""

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
from dotenv import load_dotenv

from nba_api.stats.static import players as nba_static_players

from utils.helpers import load_config, format_stat
from utils.validators import (
    ValidationError,
    validate_player_name,
)
from services.nba_fetcher import NBAFetcher
from services.predictor import StatPredictor
from services.ai_agent import AIAgent
from services.storage import StorageManager
from models.prediction import InsufficientDataError, Prediction, PredictionError
from models.player import GameLog


# Stat categories generated per prediction. Order controls the UI column order
# on the Predictor tab result card.
_PREDICTION_STATS = ("PTS", "AST", "REB")

# Official NBA logo colors. Used for chart lines and any inline brand styling
# that doesn't belong in assets/nba_theme.css.
NBA_BLUE = "#17408B"
NBA_RED = "#C9082A"
NBA_BLACK = "#000000"

# Inline-style toggles for the "Favorite this player" row, which is hidden
# until a successful search reveals it.
_FAVORITE_ROW_HIDDEN: dict = {"display": "none"}
_FAVORITE_ROW_VISIBLE: dict = {"display": "block"}


def _load_active_player_names() -> list[str]:
    """
    Build a sorted list of active NBA player names for the search datalist.

    nba_api ships a static, bundled list of players (no network call), so this
    is safe to run at module import time.
    """

    try:
        return sorted({p["full_name"] for p in nba_static_players.get_active_players() if p.get("full_name")})
    except Exception:
        return []


_ACTIVE_PLAYER_NAMES: list[str] = _load_active_player_names()


# ── Module-level setup ────────────────────────────────────────────────────────────

load_dotenv()
config = load_config("config.yaml")

storage = StorageManager(
    data_dir=config["storage"]["data_dir"],
    favorites_file=config["storage"]["favorites_file"],
    predictions_file=config["storage"]["predictions_file"],
)

predictor = StatPredictor(
    min_games=config["prediction"]["min_games_required"],
    confidence_threshold=config["prediction"]["confidence_threshold"],
)

ai_agent = AIAgent(
    model=config["openai"]["model"],
    max_tokens=config["openai"]["max_tokens"],
)

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title=config["ui"]["app_title"],
)


# ── Layout helpers ────────────────────────────────────────────────────────────────

def _build_line_chart(df: pd.DataFrame, y_col: str, title: str, color: str):
    """Build a Plotly line chart with markers and a dashed average line."""
    fig = px.line(df, x="Game", y=y_col, title=title, markers=True)
    fig.update_traces(line_color=color)
    avg_val = float(df[y_col].mean()) if not df.empty else 0.0
    fig.add_hline(
        y=avg_val,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Avg: {avg_val:.1f}",
    )
    fig.update_layout(height=config["ui"]["chart_height"])
    return fig


def _empty_chart(title: str):
    """Empty-state chart shown before any search has run."""
    fig = px.line(title=title)
    fig.update_layout(height=config["ui"]["chart_height"])
    return fig


def _build_prediction_card(player, game_log: GameLog):
    """Run PTS/AST/REB predictions + AI insight and return a Dash component.

    This is the same logic the old standalone Predictor tab used, refactored
    so the dashboard search callback can run it automatically right after
    fetching the player's recent games. Returns either a populated card or a
    `dbc.Alert` describing why the prediction couldn't be produced.

    The function intentionally swallows network/AI failures and returns a
    visible alert so a prediction error never blocks the chart render.
    """

    try:
        team_abbr = (player.team_abbreviation or "").strip().upper()
        if not team_abbr:
            return dbc.Alert(
                "Could not determine the player's current team abbreviation, "
                "so the next-game prediction was skipped.",
                color="warning",
                dismissable=True,
            )

        with NBAFetcher(season=config["nba"]["season"]) as fetcher:
            # `_detail` returns the playoff flag + series label so we can
            # badge the prediction card during the postseason.
            next_game = fetcher.get_next_game_detail(team_abbr)
            if not next_game:
                return dbc.Alert(
                    f"Could not find an upcoming game for {team_abbr}. "
                    "The team may have no scheduled games remaining.",
                    color="warning",
                    dismissable=True,
                )
            next_opponent, next_game_dt, is_playoff_game, series_label = next_game

            rest_days = int(fetcher.get_rest_days(int(player.player_id), next_game_dt))

            opp_def_ratings_map = fetcher.get_all_team_def_ratings()
            if not opp_def_ratings_map:
                return dbc.Alert(
                    "Could not load league defensive ratings from NBA.com. "
                    "Try again in a moment.",
                    color="danger",
                    dismissable=True,
                )
            opponent_def_rating_val = opp_def_ratings_map.get(next_opponent)
            if opponent_def_rating_val is None:
                return dbc.Alert(
                    f"No defensive rating available for {next_opponent}.",
                    color="danger",
                    dismissable=True,
                )
            opponent_def_rating = float(opponent_def_rating_val)

        rest_days = max(0, min(14, rest_days))

        predictions: list[Prediction] = []
        for stat_key in _PREDICTION_STATS:
            predictions.append(
                predictor.predict(
                    game_log,
                    stat=stat_key,
                    next_rest_days=rest_days,
                    next_opponent_def_rating=opponent_def_rating,
                    opp_def_ratings_map=opp_def_ratings_map,
                )
            )

        # AI insight is now always included — the toggle was removed when the
        # Predictor tab was folded into the dashboard. We still fall back to
        # the deterministic insight generator if the OpenAI call raises.
        # Pass the playoff context so the analyst-style sentence reflects
        # whether the upcoming game is a postseason matchup and whether the
        # recent-games window already includes playoff data.
        try:
            combined_insight = ai_agent.generate_insight_multi(
                game_log,
                predictions,
                rest_days,
                opponent_def_rating,
                is_playoff_game=is_playoff_game,
                playoff_series_label=series_label,
            )
        except Exception:
            combined_insight = ai_agent.generate_fallback_insight_multi(
                game_log,
                predictions,
                rest_days,
                opponent_def_rating,
                is_playoff_game=is_playoff_game,
                playoff_series_label=series_label,
            )

        for pred in predictions:
            pred.ai_insight = combined_insight
            try:
                storage.save_prediction(pred)
            except Exception:
                pass

        opponent_label = next_opponent or "Unknown (no upcoming game found)"

        stat_cols = [
            dbc.Col(
                [
                    html.Div(p.stat_category, className="text-muted small"),
                    html.H3(f"{p.predicted_value:.1f}"),
                ],
                md=4,
                className="text-center",
            )
            for p in predictions
        ]

        # Render an NBA-red "Playoffs" badge next to the title so it's
        # immediately obvious the prediction is for a postseason game.
        # `series_label` carries the round + series text from the schedule.
        title_children = [f"Next-Game Prediction — {player.name} "]
        if is_playoff_game:
            badge_text = f"🏆 PLAYOFFS · {series_label}" if series_label else "🏆 PLAYOFFS"
            title_children.append(
                dbc.Badge(
                    badge_text,
                    color="danger",
                    className="ms-2 align-middle",
                    pill=True,
                )
            )

        # Compose a small "playoff-aware" data summary so the user can see
        # the recent-games window includes the postseason. Counting here
        # (rather than passing it down separately) keeps the helper's
        # signature unchanged and works whether or not playoffs have started.
        playoff_count = sum(1 for g in game_log.games if getattr(g, "is_playoff_game", False))
        regular_count = len(game_log.games) - playoff_count
        data_mix_line = (
            f"Recent games used: {regular_count} regular season"
            + (f" + {playoff_count} playoff" if playoff_count else "")
        )

        return dbc.Card(
            dbc.CardBody(
                [
                    html.H4(title_children, className="card-title"),
                    html.Div(data_mix_line, className="text-muted small mb-3"),
                    dbc.Row(stat_cols, className="mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div("Next Opponent", className="text-muted small"),
                                    html.H5(opponent_label),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Rest Days", className="text-muted small"),
                                    html.H5(f"{rest_days}"),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        "Opponent Defense Rating",
                                        className="text-muted small",
                                    ),
                                    html.H5(
                                        f"{opponent_def_rating:.1f} "
                                        "(lower = tougher defense)"
                                    ),
                                ],
                                md=4,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.H5("AI Insight"),
                    dbc.Alert(combined_insight, color="info", className="mb-0", dismissable=True),
                ]
            ),
            className="shadow-sm",
        )

    except InsufficientDataError as e:
        return dbc.Alert(f"Not enough data for a prediction: {e}", color="warning", dismissable=True)
    except PredictionError as e:
        return dbc.Alert(f"Prediction failed: {e}", color="danger", dismissable=True)
    except Exception as e:
        return dbc.Alert(f"Could not auto-generate a prediction: {e}", color="danger", dismissable=True)


# ── Layout ────────────────────────────────────────────────────────────────────────

dashboard_tab = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Input(
                                id="player-input",
                                placeholder="Start typing a player name…",
                                type="text",
                                debounce=True,
                                list="player-suggestions",
                                autoComplete="off",
                            ),
                            # Native HTML5 datalist: the browser renders an
                            # autocomplete dropdown as the user types, filtered
                            # against the bundled active-players list. No
                            # extra Dash callback needed.
                            html.Datalist(
                                id="player-suggestions",
                                children=[
                                    html.Option(value=n) for n in _ACTIVE_PLAYER_NAMES
                                ],
                            ),
                        ],
                        md=7,
                    ),
                    dbc.Col(
                        dbc.Select(
                            id="games-select",
                            options=[
                                {"label": "5 games", "value": "5"},
                                {"label": "10 games", "value": "10"},
                                {"label": "15 games", "value": "15"},
                                {"label": "20 games", "value": "20"},
                            ],
                            value="10",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Button("Search", id="search-btn", color="primary", className="w-100"),
                        md=2,
                    ),
                ],
                className="g-2",
            ),
            html.Hr(),
            html.Div(id="player-info", className="mb-3"),
            # "Favorite this player" lets the user save the loaded player
            # to the Favorites tab. The status alert next to it confirms
            # whether the save succeeded or the player was already saved.
            # Both are hidden until a search has actually loaded a player
            # (controlled via inline `style` toggling in the search callback).
            html.Div(
                [
                    dbc.Button(
                        "⭐ Favorite this player",
                        id="favorite-btn",
                        color="secondary",
                        className="me-2",
                    ),
                    html.Span(id="favorite-status", className="ms-2"),
                ],
                id="favorite-row",
                className="mb-3",
                style={"display": "none"},
            ),
            # Auto-generated next-game prediction (AI insight included).
            # Rendered ABOVE the charts so the headline takeaway — the
            # next-game forecast — is the first thing the user sees, with
            # the supporting historical charts below.
            html.Div(id="prediction-output", className="mb-4"),
            html.Hr(),
            dcc.Graph(id="points-chart", figure=_empty_chart("Points")),
            dcc.Graph(id="assists-chart", figure=_empty_chart("Assists")),
            dcc.Graph(id="rebounds-chart", figure=_empty_chart("Rebounds")),
            dcc.Store(id="player-store"),
            dcc.Store(id="gamelog-store"),
        ]
    ),
    className="mt-3",
)


favorites_tab = dbc.Card(
    dbc.CardBody(
        [
            dbc.Button("Refresh", id="refresh-favorites-btn", color="secondary", className="mb-3"),
            html.Div(id="favorites-list"),
        ]
    ),
    className="mt-3",
)


app.layout = html.Div(
    [
        dbc.NavbarSimple(
            brand="StatEdge",
            color="primary",
            dark=True,
            fluid=True,
        ),
        dbc.Container(
            [
                dbc.Tabs(
                    [
                        dbc.Tab(dashboard_tab, label="📊 Dashboard", tab_id="tab-dashboard"),
                        dbc.Tab(favorites_tab, label="⭐ Favorites", tab_id="tab-favorites"),
                    ],
                    id="main-tabs",
                    active_tab="tab-dashboard",
                ),
            ],
            fluid=True,
            className="py-3",
        ),
    ]
)


# ── Callbacks ─────────────────────────────────────────────────────────────────────

@app.callback(
    Output("player-info", "children"),
    Output("points-chart", "figure"),
    Output("assists-chart", "figure"),
    Output("rebounds-chart", "figure"),
    Output("player-store", "data"),
    Output("gamelog-store", "data"),
    Output("prediction-output", "children"),
    Output("favorite-row", "style"),
    Output("favorite-status", "children"),
    Input("search-btn", "n_clicks"),
    State("player-input", "value"),
    State("games-select", "value"),
    prevent_initial_call=True,
)
def search_callback(n_clicks, player_name, games_value):
    """Fetch player + recent games, render charts, and auto-run the
    next-game prediction (with AI insight) so it pops up alongside the
    charts without a second button click. Also reveals the
    'Favorite this player' button for the just-loaded player."""

    try:
        if not player_name:
            return (
                dbc.Alert("Please enter a player name.", color="warning", dismissable=True),
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        clean_name = validate_player_name(player_name)
        try:
            num_games = int(games_value) if games_value is not None else 10
        except (TypeError, ValueError):
            num_games = 10

        with NBAFetcher(season=config["nba"]["season"], recent_games=num_games) as fetcher:
            player = fetcher.search_player(clean_name)
            if player is None:
                return (
                    dbc.Alert(f"No player found for '{clean_name}'.", color="danger", dismissable=True),
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )

            game_log = fetcher.get_game_log(player.player_id)
            # nba_api's PlayerGameLog no longer always includes PLAYER_NAME;
            # overwrite the log's name with the resolved player name so
            # downstream stores / AI insights show the real name.
            game_log.player_name = player.name
            season_avg = fetcher.get_season_avg(player.player_id, player.name)

        # `reversed` gives chronological order (oldest → newest) for the
        # x-axis. We carry the playoff flag through so the chart title can
        # surface how many playoff games are mixed in.
        ordered_games = list(reversed(game_log.games))
        games_df = pd.DataFrame(
            [
                {
                    "Game": i + 1,
                    "Points": g.points,
                    "Assists": g.assists,
                    "Rebounds": g.rebounds,
                }
                for i, g in enumerate(ordered_games)
            ]
        )

        playoff_games_in_window = sum(
            1 for g in ordered_games if getattr(g, "is_playoff_game", False)
        )
        # Suffix shown in chart titles whenever the recent-games window
        # actually includes playoff data — keeps regular-season views tidy.
        playoff_suffix = (
            f" (incl. {playoff_games_in_window} playoff)"
            if playoff_games_in_window
            else ""
        )

        points_fig = _build_line_chart(
            games_df,
            "Points",
            f"Points — Last {len(games_df)} Games{playoff_suffix}",
            NBA_RED,
        )
        assists_fig = _build_line_chart(
            games_df,
            "Assists",
            f"Assists — Last {len(games_df)} Games{playoff_suffix}",
            NBA_BLUE,
        )
        rebounds_fig = _build_line_chart(
            games_df,
            "Rebounds",
            f"Rebounds — Last {len(games_df)} Games{playoff_suffix}",
            NBA_BLACK,
        )

        # Build the player-info block as a two-column row: headshot on the
        # left, name + averages on the right. We fall back to no image if the
        # CDN URL is unavailable (e.g. missing player_id).
        headshot_url = player.headshot_url
        headshot_col = (
            dbc.Col(
                html.Img(
                    src=headshot_url,
                    alt=f"{player.name} headshot",
                    style={
                        "width": "100%",
                        "maxWidth": "160px",
                        "borderRadius": "8px",
                        "objectFit": "cover",
                    },
                ),
                md="auto",
                className="pe-3",
            )
            if headshot_url
            else None
        )

        info_text_col = dbc.Col(
            [
                html.H4(player.display_name, className="mb-1"),
                html.Div(
                    [
                        html.Strong("Season Averages: "),
                        html.Span(season_avg.summary),
                    ]
                ),
                html.Small(
                    (
                        f"Showing last {format_stat(len(games_df), 0)} games"
                        + (
                            f" — {playoff_games_in_window} from the playoffs"
                            if playoff_games_in_window
                            else ""
                        )
                    ),
                    className="text-muted",
                ),
            ],
            md=True,
        )

        info = dbc.Row(
            [c for c in (headshot_col, info_text_col) if c is not None],
            className="align-items-center g-0",
        )

        player_store = {
            "player_id": player.player_id,
            "name": player.name,
            "team": player.team,
            "position": player.position,
            "team_abbreviation": player.team_abbreviation,
        }

        gamelog_store = {
            "player_name": game_log.player_name,
            "games": [
                {
                    "player_name": g.player_name,
                    "points": g.points,
                    "assists": g.assists,
                    "rebounds": g.rebounds,
                    "fg_percentage": g.fg_percentage,
                    "fg3_percentage": g.fg3_percentage,
                    "games_played": g.games_played,
                    "is_season_average": g.is_season_average,
                    "game_date": g.game_date,
                    "opponent_abbreviation": g.opponent_abbreviation,
                    "is_playoff_game": g.is_playoff_game,
                }
                for g in game_log.games
            ],
        }

        # Auto-run the next-game prediction so the AI insight pops up right
        # below the charts. Failures inside this helper are returned as
        # alerts so they never break the chart render.
        prediction_card = _build_prediction_card(player, game_log)

        return (
            info,
            points_fig,
            assists_fig,
            rebounds_fig,
            player_store,
            gamelog_store,
            prediction_card,
            _FAVORITE_ROW_VISIBLE,
            "",
        )

    except ValidationError as e:
        return (
            dbc.Alert(f"Input error: {e}", color="warning", dismissable=True),
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )
    except Exception as e:
        return (
            dbc.Alert(f"Something went wrong: {e}", color="danger", dismissable=True),
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )


def _render_favorites_list():
    """Read favorites from disk and render as a Bootstrap list group.

    Each item gets a small "✕" remove button that removes that
    favorite via the pattern-matching `remove_favorite_callback`. The
    button id is dict-shaped so Dash can fan callbacks out to all of
    them at once.
    """

    try:
        favorites = storage.load_favorites()
    except Exception as e:
        return dbc.Alert(f"Failed to load favorites: {e}", color="danger", dismissable=True)

    if not favorites:
        return dbc.Alert(
            "No favorites saved yet. Save a player from the Dashboard tab.",
            color="info",
            dismissable=True,
        )

    items = []
    for fav in favorites:
        name = fav.get("name", "Unknown")
        team = fav.get("team", "Unknown")
        position = fav.get("position", "?")
        # Each per-row button uses a pattern-matching dict id so a single
        # callback can fan out to all of them. `index` carries the player
        # name we need to act on.
        view_btn = dbc.Button(
            "View on Dashboard",
            id={"type": "load-favorite-btn", "index": name},
            color="primary",
            size="sm",
            className="ms-auto",
            title=f"Open {name}'s charts and prediction on the Dashboard",
        )
        remove_btn = dbc.Button(
            "✕",
            id={"type": "remove-favorite-btn", "index": name},
            color="danger",
            outline=True,
            size="sm",
            className="ms-2",
            title=f"Remove {name} from favorites",
        )
        items.append(
            dbc.ListGroupItem(
                [
                    html.Strong(name),
                    html.Span(f" — {team} ({position})", className="text-muted ms-2"),
                    view_btn,
                    remove_btn,
                ],
                className="d-flex align-items-center",
            )
        )

    return dbc.ListGroup(items, flush=True)


@app.callback(
    Output("favorites-list", "children"),
    Output("favorite-status", "children", allow_duplicate=True),
    Input("refresh-favorites-btn", "n_clicks"),
    Input("favorite-btn", "n_clicks"),
    Input({"type": "remove-favorite-btn", "index": ALL}, "n_clicks"),
    State("player-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def favorites_callback(_refresh_clicks, favorite_clicks, remove_clicks, player_data):
    """Single source of truth for the favorites JSON file + list UI.

    Handles three triggers:
      - Refresh button → just re-render the list.
      - "⭐ Favorite this player" button on the dashboard → save the
        currently-loaded player to favorites and show a status badge
        next to the button.
      - Per-item "✕" remove button on the Favorites tab → remove that
        player from the JSON file.
    Doing all writes inside one callback avoids races between the
    JSON write and the list re-render.
    """

    triggered = ctx.triggered_id
    status_msg = no_update

    # ── Add (dashboard "Favorite this player" button) ─────────────
    if triggered == "favorite-btn" and favorite_clicks:
        if not player_data or not player_data.get("name"):
            status_msg = dbc.Badge(
                "Search for a player first.", color="warning", className="ms-2"
            )
        else:
            try:
                added = storage.save_favorite(
                    {
                        "name": player_data.get("name"),
                        "team": player_data.get("team"),
                        "position": player_data.get("position"),
                        "team_abbreviation": player_data.get("team_abbreviation"),
                        "player_id": player_data.get("player_id"),
                    }
                )
                status_msg = dbc.Badge(
                    f"Added {player_data['name']} to favorites!"
                    if added
                    else f"{player_data['name']} is already in favorites.",
                    color="success" if added else "secondary",
                    className="ms-2",
                )
            except Exception as e:
                status_msg = dbc.Badge(
                    f"Save failed: {e}", color="danger", className="ms-2"
                )

    # ── Remove (per-item "✕" on the Favorites tab) ────────────────
    elif (
        isinstance(triggered, dict)
        and triggered.get("type") == "remove-favorite-btn"
        and remove_clicks
        and any(remove_clicks)
    ):
        target_name = triggered.get("index")
        if target_name:
            try:
                storage.remove_favorite(target_name)
            except Exception:
                # Best effort — re-render still works even if delete fails.
                pass

    return _render_favorites_list(), status_msg


@app.callback(
    Output("player-input", "value"),
    Output("main-tabs", "active_tab"),
    Output("search-btn", "n_clicks"),
    Input({"type": "load-favorite-btn", "index": ALL}, "n_clicks"),
    State("search-btn", "n_clicks"),
    prevent_initial_call=True,
)
def load_favorite_callback(n_clicks_list, current_search_clicks):
    """Open a favorited player's Dashboard view in one click.

    When the user clicks "View on Dashboard" on any favorite item, we:
      1. Drop their name into the search input.
      2. Switch the active tab back to the Dashboard.
      3. Programmatically bump the Search button's n_clicks, which is
         enough to re-fire `search_callback` and re-generate the player
         info, charts, and AI prediction — exactly as if the user had
         typed the name and clicked Search themselves.
    Bumping n_clicks (instead of duplicating search logic here) keeps
    one code path responsible for rendering a player.
    """

    if not ctx.triggered_id or not any(n_clicks_list or []):
        return no_update, no_update, no_update

    try:
        target_name = ctx.triggered_id.get("index")
    except AttributeError:
        return no_update, no_update, no_update

    if not target_name:
        return no_update, no_update, no_update

    next_clicks = (current_search_clicks or 0) + 1
    return target_name, "tab-dashboard", next_clicks


# ── Entrypoint ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)
