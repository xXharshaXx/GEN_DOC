import os
import json
from typing import List, Dict, Any
from .config import settings


MAX_TURNS = 20


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _session_path(session_id: str) -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), settings.chat_history_dir)
    _ensure_dir(base)
    return os.path.join(base, f"{session_id}.txt")


def load_history(session_id: str) -> List[Dict[str, Any]]:
    path = _session_path(session_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            lines = [json.loads(line) for line in f if line.strip()]
            return lines[-MAX_TURNS:]
        except Exception:
            return []


def append_turn(session_id: str, role: str, content: str) -> None:
    path = _session_path(session_id)
    turns = load_history(session_id)
    turns.append({"role": role, "content": content})
    # keep only last MAX_TURNS
    turns = turns[-MAX_TURNS:]
    with open(path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


