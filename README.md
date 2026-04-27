# 🏀 NBA Analytics App

A full-stack Python analytics dashboard that lets users search NBA players, visualize their performance trends, and get AI-powered stat predictions for upcoming games.

Built with **Streamlit**, **nba_api**, **scikit-learn**, and **OpenAI**.

---

## 📌 Project Overview

This app has two core modules:

- **Dashboard** — Search any NBA player and view their season stats with interactive charts
- **Predictor** — Select a player and get a machine learning + AI-powered prediction of their next game stats

---

## 🗂️ Project Structure

```
nba-analytics-app/
│
├── app.py                  # Streamlit entry point — runs the entire app
│
├── models/
│   ├── __init__.py
│   ├── player.py           # Player and PlayerStats dataclasses (OOP)
│   └── prediction.py       # Prediction dataclass + custom exceptions
│
├── services/
│   ├── __init__.py
│   ├── nba_fetcher.py      # Pulls and cleans data from nba_api
│   ├── predictor.py        # scikit-learn ML prediction engine
│   ├── ai_agent.py         # OpenAI natural language insights
│   └── storage.py          # JSON file storage (save/load favorites)
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py          # Shared formatting and utility functions
│   └── validators.py       # Input validation functions
│
├── tests/
│   ├── __init__.py
│   ├── test_models.py      # Unit tests for Player and Prediction classes
│   └── test_services.py    # Unit tests for services (mocked API calls)
│
├── data/                   # Auto-created folder for saved JSON data
│
├── config.yaml             # App-wide settings (season, model config, etc.)
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
├── .gitignore              # Files to exclude from Git
└── README.md               # This file
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/nba-analytics-app.git
cd nba-analytics-app
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
cp .env.example .env
# Open .env and add your OpenAI API key
```

### 5. Run the App
```bash
streamlit run app.py
```

---

## ☁️ Deploy to Railway

The repo ships with a `Procfile`, `railway.json`, and `.python-version`, so Railway will auto-detect, build, and serve the app with Gunicorn.

### Option A — Deploy from the Railway dashboard (easiest)

1. Push this repo to GitHub (already done if you cloned it).
2. Go to [railway.com](https://railway.com) → **New Project** → **Deploy from GitHub repo**.
3. Pick this repository and let Railway run the first build (it will use Nixpacks + `requirements.txt`).
4. Open the new service → **Variables** → add:
   - `OPENAI_API_KEY` = your real OpenAI key
5. Open the **Settings** tab → **Networking** → **Generate Domain** to get a public URL.
6. Wait for the deploy to go green and visit the domain. Done.

### Option B — Deploy from the Railway CLI

```bash
npm i -g @railway/cli
railway login
railway init                # link this folder to a new Railway project
railway variables --set OPENAI_API_KEY=sk-...
railway up                  # build & deploy
railway domain              # generate a public URL
```

### Notes for production

- Gunicorn is started via the `Procfile`:
  `gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
  Single worker + threads avoids duplicating the in-memory caches; the long timeout gives `nba_api` room to respond.
- Railway's filesystem is **ephemeral** — `data/favorites.json` and `data/predictions.json` reset on every redeploy. If you want them to persist, add a Railway **Volume** mounted at `/app/data` from the service's **Settings** tab.
- Python version is pinned to `3.12.3` via `.python-version` to match the `numpy 1.26.4` / `pandas 2.2.2` wheels in `requirements.txt`.

> ⚠️ **NBA API + cloud hosts:** `stats.nba.com` actively throttles requests from Railway/Render/Heroku/Fly IP ranges, which causes read-timeout errors that don't reproduce locally. If you hit this, **deploy to PythonAnywhere instead** (see below) — `stats.nba.com` is on PythonAnywhere's outbound whitelist for free accounts.

---

## ☁️ Deploy to PythonAnywhere (recommended for `nba_api`)

PythonAnywhere doesn't block requests to `stats.nba.com`, so this is the most reliable free host for an `nba_api`-driven app. The deploy is a bit more manual than Railway because PythonAnywhere uses a WSGI config file instead of a Procfile — but the repo includes `pythonanywhere_wsgi.py` as a copy-paste template.

### One-time setup

1. **Sign up** at [pythonanywhere.com](https://www.pythonanywhere.com) and pick the free Beginner plan.
2. **Open a Bash console** (Dashboard → "New console" → Bash).
3. **Clone the repo**:
   ```bash
   git clone https://github.com/charles-sharp29/Sharp-CS302-Final-Project.git NBA_Analytics_Final_Project
   cd NBA_Analytics_Final_Project
   ```
4. **Create a virtualenv with Python 3.11** that inherits PythonAnywhere's pre-installed scientific stack:
   ```bash
   mkvirtualenv --python=python3.11 --system-site-packages statedge-venv
   pip install --no-cache-dir -r requirements-pythonanywhere.txt
   ```
   This takes about a minute. **Do not use Python 3.12 on the free tier** — PythonAnywhere only ships `pandas` / `numpy` / `scikit-learn` / `plotly` / `dash` as pre-installed system packages on Python 3.10 and 3.11, so on 3.12 you'd have to install all of them yourself and blow past the 512 MB quota. `--system-site-packages` lets the venv re-use the system copies, and `requirements-pythonanywhere.txt` only pins the ~5 small packages PythonAnywhere doesn't already include.
5. **Create the `.env` file** with your OpenAI key:
   ```bash
   echo "OPENAI_API_KEY=sk-your-real-key-here" > .env
   ```

### Configure the web app

6. Go to the **Web** tab → **Add a new web app** → **Manual configuration** (NOT Flask) → pick **Python 3.11** (must match the venv).
7. On the resulting Web tab, set:
   - **Source code:** `/home/<YOUR_USERNAME>/NBA_Analytics_Final_Project`
   - **Working directory:** `/home/<YOUR_USERNAME>/NBA_Analytics_Final_Project`
   - **Virtualenv:** `/home/<YOUR_USERNAME>/.virtualenvs/statedge-venv`
8. Click the **WSGI configuration file** link near the top of the Web tab. Replace its entire contents with the contents of `pythonanywhere_wsgi.py` from this repo, then change `<YOUR_USERNAME>` to your real PythonAnywhere username. Save. The proxy env vars at the top of that file are required — without them outbound NBA / OpenAI requests get blocked by PythonAnywhere's free-tier outbound firewall and time out at 30s.
9. Click the green **Reload** button at the top of the Web tab.
10. Visit `https://<YOUR_USERNAME>.pythonanywhere.com` — done.

### Updating later

```bash
workon statedge-venv
cd ~/NBA_Analytics_Final_Project
git pull
# if requirements-pythonanywhere.txt changed:
pip install --no-cache-dir -r requirements-pythonanywhere.txt
```
Then click **Reload** on the Web tab.

### Recovering from a "disk quota exceeded" install

If `pip install` errors with `[Errno 122] Disk quota exceeded`, your venv is too big — almost always because the venv was created on Python 3.12 (no system batteries available) or without `--system-site-packages`. Clean up and recreate on 3.11:

```bash
deactivate 2>/dev/null || true
rmvirtualenv statedge-venv
rm -rf ~/.cache/pip
mkvirtualenv --python=python3.11 --system-site-packages statedge-venv
pip install --no-cache-dir -r requirements-pythonanywhere.txt
```

Then check usage with `du -sh ~/.virtualenvs/statedge-venv ~/NBA_Analytics_Final_Project ~/.cache 2>/dev/null` — a healthy venv built this way is only ~30–60 MB, well under quota.

### Free-tier limits to know about

- **CPU:** 100 seconds/day of CPU time. The first NBA API call after a cold start can be slow, but a typical session uses far less than this.
- **Disk:** 512 MB. A fresh install of this app's dependencies sits around ~300 MB, so there's headroom but don't add anything large.
- **Idle expiration:** Free web apps that get no traffic for 1 month go offline — visit your URL once a month to keep it alive, or click Reload.
- **Outbound whitelist:** `stats.nba.com` and `api.openai.com` are both already whitelisted; this app doesn't talk to anything else, so no allowlist requests are needed.

---

## 🧪 Running Tests
```bash
pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Layer        | Tool              | Purpose                        |
|--------------|-------------------|--------------------------------|
| UI           | Streamlit         | Web dashboard interface        |
| Data         | nba_api + pandas  | Real NBA player/game data      |
| ML           | scikit-learn      | Stat prediction model          |
| AI Insights  | OpenAI API        | Natural language predictions   |
| Storage      | JSON files        | Save favorite players          |
| Testing      | pytest            | Unit tests                     |
| Config       | PyYAML            | App-wide settings              |

---

## 🎯 Features

- 🔍 Search any active NBA player by name
- 📊 View points, assists, rebounds trends over last N games
- 🤖 ML model predicts next-game stat totals
- 💬 OpenAI generates a readable insight about the prediction
- ⭐ Save and manage favorite players (persistent JSON storage)
- ✅ Full input validation and error handling throughout

---

## 📋 Key Design Decisions

- **Streamlit over React** — Pure Python stack keeps complexity manageable and is perfectly suited for data dashboards
- **JSON storage over a database** — Lightweight persistence without requiring PostgreSQL setup
- **Dataclasses for OOP** — Python's `@dataclass` decorator provides clean, modern OOP with built-in validation
- **Service layer separation** — Fetching, prediction, and AI are all separate services so each can be tested independently

---

## 👨‍💻 Author

Built as a final project for Programming 2 — demonstrating SDLC, OOP, testing, and AI integration.
