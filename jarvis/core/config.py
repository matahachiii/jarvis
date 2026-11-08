"""
Configuration loader for Jarvis.

Reads config/config.toml (settings) and .env (secrets) once,
so the rest of the code has one single place to get settings from.
"""

import tomllib
from pathlib import Path

from dotenv import load_dotenv

# Project root = two folders up from this file (jarvis/core/config.py -> Jarvis/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.toml"
ENV_PATH = PROJECT_ROOT / ".env"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load secrets from .env and settings from config.toml. Returns the settings as a dict."""
    # Secrets from .env become environment variables (readable via os.environ)
    load_dotenv(ENV_PATH)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        return tomllib.load(f)
