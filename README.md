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
