import json
import os

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "run_history.json")


def load(n_recent: int = None) -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE) as f:
        history = json.load(f)
    return history[-n_recent:] if n_recent else history


def save(entry: dict):
    history = load()
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
