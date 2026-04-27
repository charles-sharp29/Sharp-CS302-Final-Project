"""
pythonanywhere_wsgi.py
======================
Reference WSGI configuration for deploying StatEdge on PythonAnywhere.

PythonAnywhere does NOT auto-detect Procfile / railway.json. Instead it uses a
WSGI file located at /var/www/<username>_pythonanywhere_com_wsgi.py that
exposes a top-level `application` callable.

Copy the body of this file into that WSGI file (replacing whatever default
PythonAnywhere put there), update <YOUR_USERNAME>, then click Reload on the
Web tab.
"""

import os
import sys


# ── 1. Add the project to sys.path so `import app` works ────────────────────────
# Replace <YOUR_USERNAME> with your actual PythonAnywhere username.
PROJECT_HOME = "/home/<YOUR_USERNAME>/NBA_Analytics_Final_Project"

if PROJECT_HOME not in sys.path:
    sys.path.insert(0, PROJECT_HOME)


# ── 2. Load .env so OPENAI_API_KEY (and anything else) is available ─────────────
# python-dotenv is already a dependency of this project. The .env file lives at
# the project root and is NOT committed to git, so you'll create it manually
# on PythonAnywhere via the Files tab or a Bash console:
#     echo "OPENAI_API_KEY=sk-..." > /home/<YOUR_USERNAME>/NBA_Analytics_Final_Project/.env
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(PROJECT_HOME, ".env"))


# ── 3. Import the Flask server that Dash exposes via app.server ─────────────────
# `app.py` sets `server = app.server` at module top level specifically so WSGI
# hosts can pick it up here. PythonAnywhere expects the WSGI callable to be
# named `application`.
from app import server as application  # noqa: E402, F401
