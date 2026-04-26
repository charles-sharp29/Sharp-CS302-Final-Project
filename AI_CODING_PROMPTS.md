# 🤖 AI Coding Prompts — NBA Analytics App
# ============================================================
# HOW TO USE THIS FILE:
#
# Each section below is a self-contained prompt you paste directly
# into any AI coding assistant (ChatGPT, Copilot, Gemini, etc.).
#
# TIPS FOR BEST RESULTS:
#   1. Paste ONE prompt at a time — don't combine them
#   2. After the AI responds, say: "Now write the complete file
#      with no placeholders, no TODOs, and no omitted sections"
#   3. If the AI skips something, say: "Show the full [function name]
#      implementation — do not summarize or truncate"
#   4. Always ask: "Are there any import statements I'm missing?"
#   5. If you get an error running the code, paste the full error
#      message back and say: "Fix this error in the file you just wrote"
#   6. After all files are done, say: "Review all files for
#      consistency — make sure imports match what was actually defined"
# ============================================================


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT 1 — models/player.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I am building an NBA analytics Streamlit app in Python. Write the complete
file `models/player.py`. This file defines the core data structures.

Requirements:
- Use Python @dataclass decorator for all three classes
- Class 1: Player — fields: name (str), team (str), position (str),
  player_id (Optional[int] = None). Include __post_init__ that validates
  name is not empty and applies .strip().title(). Include a @property
  called display_name that returns "{name} ({position}) — {team}".
- Class 2: PlayerStats — fields: player_name (str), points (float=0.0),
  assists (float=0.0), rebounds (float=0.0), fg_percentage (float=0.0),
  fg3_percentage (float=0.0), games_played (int=0), is_season_average
  (bool=False). Include __post_init__ that raises ValueError for negative
  stats. Include a @property called summary returning a formatted string.
- Class 3: GameLog — fields: player_name (str), games (List[PlayerStats],
  default_factory=list). Include @property for average_points,
  average_assists, average_rebounds. Include game_count() method.
- Add type hints on every method and field
- Add a docstring to every class and method explaining its purpose
- Do not use any external libraries — only Python stdlib (dataclasses,
  typing)

Write the complete file with no omissions.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT 2 — models/prediction.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I am building an NBA analytics Streamlit app in Python. Write the complete
file `models/prediction.py`. This file defines the Prediction data
structure and custom exceptions.

Requirements:
- Custom exception class PredictionError(Exception) with docstring
- Custom exception class InsufficientDataError(PredictionError) —
  inherits from PredictionError, with docstring
- @dataclass class Prediction with fields: player_name (str),
  stat_category (str), predicted_value (float), confidence (float),
  ai_insight (str = ""), created_at (str, auto-set using field
  default_factory to datetime.now().strftime("%Y-%m-%d %H:%M"))
- __post_init__ that raises PredictionError if predicted_value < 0
  or if confidence is not between 0.0 and 1.0
- @property confidence_label: returns "High" if >= 0.8, "Medium" if
  >= 0.6, else "Low"
- @property formatted_result: returns a human-readable prediction string
- Method to_dict() that returns all fields as a plain dictionary
  (used for JSON storage)
- Type hints and docstrings on everything
- Only use stdlib imports: dataclasses, datetime, typing

Write the complete file with no omissions.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT 3 — utils/helpers.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I am building an NBA analytics Streamlit app in Python. Write the complete
file `utils/helpers.py`. This file contains shared utility functions.

Requirements:
- safe_float(value, default=0.0): converts any value to float safely,
  returns default on failure
- safe_int(value, default=0): converts any value to int safely, returns
  default on failure
- format_stat(value: float, decimal_places: int = 1) -> str: formats a
  float to N decimal places as string
- format_percentage(value: float) -> str: converts decimal (0.523) to
  "52.3%"
- load_config(config_path: str = "config.yaml") -> dict: reads a YAML
  file using PyYAML and returns as dict. Raises FileNotFoundError with a
  clear message if missing. Uses pathlib.Path for the file path.
- get_stat_color(value: float, avg: float) -> str: returns "#28a745"
  if value > avg*1.1, "#dc3545" if value < avg*0.9, else "#6c757d"
- Type hints and docstrings on all functions
- Imports needed: yaml, pathlib.Path, typing.Any

Write the complete file with no omissions.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT 4 — utils/validators.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I am building an NBA analytics Streamlit app in Python. Write the complete
file `utils/validators.py`. This file validates all user inputs.

Requirements:
- Custom exception: ValidationError(ValueError)
- Constants: VALID_STATS = {"PTS", "AST", "REB"}, MIN_GAMES = 5,
  MAX_GAMES = 82
- validate_player_name(name: str) -> str:
    - Raises ValidationError if not a string, if empty after strip,
      if length < 2, or if name contains any digits
    - Returns name.strip().title() if valid
- validate_stat_category(stat: str) -> str:
    - Strips and uppercases input
    - Raises ValidationError if not in VALID_STATS
    - Returns the uppercased stat string
- validate_game_count(count: int) -> int:
    - Converts to int, raises ValidationError if it fails
    - Raises ValidationError if count < MIN_GAMES or > MAX_GAMES
    - Returns the integer
- Type hints and docstrings on all functions
- No external imports needed

Write the complete file with no omissions.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT 5 — services/nba_fetcher.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I am building an NBA analytics Streamlit app in Python. Write the complete
file `services/nba_fetcher.py`. This file fetches NBA data using nba_api.

Requirements:
- Class NBAFetcher with __init__(self, season="2024-25", recent_games=10)
- Implement as a context manager: __enter__ returns self, __exit__
  returns False (does not suppress exceptions)
- self._request_delay = 0.6 (seconds between API calls to avoid
  rate limiting — use time.sleep() before each API call)
- search_player(name: str) -> Optional[Player]:
    - Uses nba_api.stats.static.players.find_players_by_full_name(name)
    - Returns None if no results
    - Fetches commonplayerinfo for team and position
    - Returns a Player object (from models.player)
    - Wraps in try/except, raises RuntimeError with clear message on fail
- get_game_log(player_id: int) -> GameLog:
    - Uses nba_api.stats.endpoints.playergamelog.PlayerGameLog
    - Gets last self.recent_games rows
    - Uses a private generator method _parse_game_rows(df) that yields
      one PlayerStats per row (this is the generator/yield feature)
    - Raises RuntimeError on failure
- get_season_avg(player_id: int, player_name: str) -> PlayerStats:
    - Uses nba_api.stats.endpoints.playercareerstats.PlayerCareerStats
    - Filters to current season, falls back to most recent if missing
    - Returns PlayerStats with is_season_average=True
- _parse_game_rows(df) is a Generator[PlayerStats, None, None]
  using yield — call safe_float() from utils.helpers on all values
- Imports: time, typing, nba_api endpoints, pandas, models.player,
  utils.helpers.safe_float and safe_int

Write the complete file with no omissions.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT 6 — services/predictor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I am building an NBA analytics Streamlit app in Python. Write the complete
file `services/predictor.py`. This is the machine learning prediction engine.

Requirements:
- Class StatPredictor with __init__(self, min_games=5,
  confidence_threshold=0.6). Private attributes: self._model = None,
  self._trained_stat = None
- predict(game_log: GameLog, stat: str = "PTS") -> Prediction:
    - Raises InsufficientDataError if game_log.game_count() < min_games
    - Calls _extract_stat_values() to get a list of floats
    - Calls _train() to fit the model and get confidence (R² score)
    - Calls _forecast() to get the next predicted value
    - Clamps result to >= 0.0 with max(0.0, predicted_value)
    - Returns a Prediction object (from models.prediction)
- evaluate(game_log: GameLog, stat: str = "PTS") -> float:
    - Trains the model and returns the R² score as float
- _train(y_values: list) -> float (private):
    - X = numpy arange(1, len+1).reshape(-1, 1)
    - Fits sklearn LinearRegression
    - Returns r2_score(y, y_pred), clamped to >= 0.0
- _forecast(next_index: int) -> float (private):
    - Raises PredictionError if self._model is None
    - Predicts and returns float
- _extract_stat_values(game_log, stat) -> list (private):
    - stat_map maps "PTS"->points, "AST"->assists, "REB"->rebounds
    - Raises PredictionError for unknown stat
    - Returns list in chronological order (oldest first using reversed())
- Imports: numpy, sklearn.linear_model.LinearRegression,
  sklearn.metrics.r2_score, models.player.GameLog,
  models.prediction.Prediction/PredictionError/InsufficientDataError

Write the complete file with no omissions.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT 7 — services/ai_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I am building an NBA analytics Streamlit app in Python. Write the complete
file `services/ai_agent.py`. This file integrates OpenAI for natural
language stat insights.

Requirements:
- A retry decorator factory: retry(max_attempts=3, delay=2.0) that
  returns a decorator. The decorator wraps a function in a loop: tries
  up to max_attempts times, sleeps delay seconds between failures, raises
  RuntimeError with last error on final failure. This is the decorator
  pattern.
- Class AIAgent with __init__(self, model="gpt-3.5-turbo",
  max_tokens=150). Loads OpenAI API key from env using os.environ.get
  and creates self._client = OpenAI(api_key=...). Call load_dotenv()
  at module level.
- generate_insight(self, game_log: GameLog, prediction: Prediction) -> str:
    - Decorated with @retry(max_attempts=3, delay=2.0)
    - Calls _build_prompt() to create the prompt string
    - Sends to self._client.chat.completions.create with a system message
      that says: "You are a concise NBA sports analyst. Give 2-3 sentence
      insights about a player's upcoming performance based on their recent
      stats. Be specific and use the numbers provided."
    - Returns response.choices[0].message.content.strip()
- generate_fallback_insight(self, game_log, prediction) -> str:
    - Returns a manually formatted string using averages from game_log
      and predicted_value from prediction. No API call.
- _build_prompt(self, game_log, prediction) -> str (private):
    - Includes player name, last 5 game values for PTS/AST/REB,
      season averages, and the ML prediction with confidence label
- Imports: os, time, openai.OpenAI, dotenv.load_dotenv,
  models.player.GameLog, models.prediction.Prediction

Write the complete file with no omissions.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT 8 — services/storage.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I am building an NBA analytics Streamlit app in Python. Write the complete
file `services/storage.py`. This handles saving/loading JSON data files.

Requirements:
- Class StorageManager with __init__(self, data_dir="data",
  favorites_file="favorites.json", predictions_file="predictions.json")
    - self.data_dir = Path(data_dir)
    - Creates the data directory using mkdir(parents=True, exist_ok=True)
- save_favorite(player_dict: dict) -> bool:
    - Loads existing favorites, checks for duplicate by name (case-
      insensitive), appends if not duplicate, writes back
    - Returns True if saved, False if duplicate
- load_favorites() -> List[dict]:
    - Returns list from JSON file, or [] if file missing
- remove_favorite(player_name: str) -> bool:
    - Filters out player by name (case-insensitive), writes back
    - Returns True if removed, False if not found
- save_prediction(prediction: Prediction) -> None:
    - Appends prediction.to_dict() to predictions JSON file
- load_predictions(player_name: Optional[str] = None) -> List[dict]:
    - Returns all predictions, or filtered by player_name if provided
- Private _read_json(path: Path, default) -> any:
    - Uses 'with open(path, "r") as f' context manager
    - Returns json.load(f) or default on FileNotFoundError/JSONDecodeError
- Private _write_json(path: Path, data) -> None:
    - Uses 'with open(path, "w") as f' context manager
    - Writes json.dump(data, f, indent=2)
- Imports: json, pathlib.Path, typing, models.prediction.Prediction

Write the complete file with no omissions.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT 9 — tests/test_models.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I am building an NBA analytics Streamlit app in Python. Write the complete
file `tests/test_models.py` using pytest.

Requirements — write tests for ALL of the following:
- Player: valid creation, name normalization (.strip().title()), empty
  name raises ValueError, display_name property format
- PlayerStats: valid creation, summary property includes PTS/AST/REB,
  negative points raises ValueError, zero stats are valid
- GameLog: average_points returns correct mean, empty log returns 0.0,
  game_count() returns correct number
- Prediction: valid creation, confidence_label returns "High"/"Medium"/"Low"
  based on value ranges, negative predicted_value raises PredictionError,
  confidence > 1.0 raises PredictionError, to_dict() has required keys,
  formatted_result contains player name
- Use pytest.raises() for all exception tests
- Use pytest.approx() for float comparisons
- Organize tests into classes (TestPlayer, TestPlayerStats, etc.)
- At minimum 10 test functions total
- No mocking needed — these test pure logic

Write the complete file with no omissions.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT 10 — tests/test_services.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I am building an NBA analytics Streamlit app in Python. Write the complete
file `tests/test_services.py` using pytest.

Requirements — write tests for ALL of the following:

Helper function at top: make_game_log(num_games=10, points=25.0) that
returns a GameLog with num_games PlayerStats objects (increments points
by i for each game).

StatPredictor tests:
- predict() returns a Prediction object for 10-game log
- predicted_value is >= 0
- InsufficientDataError raised for 3-game log (min=5)
- PredictionError raised for stat="BLOCKS"
- predict() works for stat="AST"

StorageManager tests (use pytest tmp_path fixture for temp files):
- save_favorite() saves and load_favorites() retrieves it
- Duplicate save returns False and doesn't create duplicate
- remove_favorite() removes the player
- remove_favorite() for nonexistent player returns False
- load_favorites() on fresh storage returns []

Validator tests:
- validate_player_name() returns cleaned name for valid input
- validate_player_name() raises ValidationError for empty string
- validate_player_name() raises ValidationError for name with digits
- validate_stat_category() accepts "pts", "AST", "reb" (case insensitive)
- validate_stat_category() raises ValidationError for "BLOCKS"
- validate_game_count() returns int for valid input
- validate_game_count() raises ValidationError for count=2 (below min)
- validate_game_count() raises ValidationError for count=100 (above max)

Use pytest.raises() for all exception tests. No real API calls —
use pytest tmp_path fixture for storage tests. Minimum 15 test functions.

Write the complete file with no omissions.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROMPT 11 — app.py (Streamlit UI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I am building an NBA analytics Streamlit app in Python. Write the complete
file `app.py`. This is the Streamlit entry point that runs with
`streamlit run app.py`.

Requirements:
- Load config.yaml at startup using load_config() from utils.helpers
- Set page config: title from config["ui"]["app_title"], page_icon="🏀",
  layout="wide"
- Initialize st.session_state keys: current_player=None,
  current_game_log=None, last_prediction=None
- Instantiate at module level: StorageManager, StatPredictor, AIAgent
  (all using values from config)
- Sidebar with st.radio navigation: "📊 Dashboard", "🤖 Predictor",
  "⭐ Favorites". Show season from config as caption.

Dashboard page:
  - Text input for player name + slider for num_games (5-20)
  - On Search button click: validate with validate_player_name(), use
    NBAFetcher as context manager to get player, game_log, season_avg
  - Show player.display_name and season_avg.summary on success
  - "Save to Favorites" button calls storage.save_favorite()
  - Show 3 Plotly line charts in tabs (Points, Assists, Rebounds) with
    a dashed average line on each
  - Catch ValidationError and Exception separately with st.error()

Predictor page:
  - If session_state.current_player is None, show st.info() to go to
    Dashboard first
  - Show player name, selectbox for stat (PTS/AST/REB with friendly labels)
  - Checkbox: "Include AI insight"
  - On Predict button: call predictor.predict(), then ai_agent.
    generate_insight() (fall back to generate_fallback_insight on error)
  - Show 3 st.metric columns: predicted value, confidence label, R² score
  - Show AI insight in st.info()
  - Catch InsufficientDataError, PredictionError, Exception separately

Favorites page:
  - Load and display all favorites with name/team/position
  - Remove button per player that calls storage.remove_favorite() and
    st.rerun()

Imports needed: streamlit, plotly.express, pandas, dotenv.load_dotenv,
all service and model classes.

Write the complete file with no omissions. Do not use HTML <form> tags.
Use only Streamlit components for all UI elements.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL CONSISTENCY CHECK PROMPT (run this last)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I have a Python NBA analytics Streamlit app. I need you to review all
my files for consistency and catch any issues before I run the app.

The project structure is:
  app.py, models/player.py, models/prediction.py, services/nba_fetcher.py,
  services/predictor.py, services/ai_agent.py, services/storage.py,
  utils/helpers.py, utils/validators.py, tests/test_models.py,
  tests/test_services.py

Please check for:
1. Import mismatches (e.g. a file imports something that doesn't exist)
2. Function names that are called differently than they are defined
3. Missing __init__.py files in models/, services/, utils/, tests/
4. Any function that is called with the wrong number of arguments
5. Any class that is instantiated but never imported in the calling file

For each issue found, show me: which file has the bug, what the bug is,
and the corrected line(s). If everything looks consistent, confirm that.

[Paste all your file contents after this prompt]
