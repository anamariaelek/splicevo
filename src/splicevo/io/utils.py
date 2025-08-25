import json
import pickle
from pathlib import Path
from typing import Any


def save_json(obj: Any, path: str | Path, indent: int = 4) -> None:
    """Save an object as a JSON file."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def load_json(path: str | Path) -> Any:
    """Load an object from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save an object to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    """Load an object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
