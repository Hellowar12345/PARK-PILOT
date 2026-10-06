"""
SQLite trace logger — 每個 thinking / tool_call / state_change 都寫一筆，
demo 結束可一鍵「決策回放」，也是「我能解釋為什麼這樣決策」的最強證據。
"""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = str(Path(__file__).parent.parent / "agent_traces.db")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.execute(
        """CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            mission_id TEXT,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL
        )"""
    )
    return c


def log(kind: str, payload: dict[str, Any], mission_id: str = "") -> None:
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO traces (ts, mission_id, kind, payload) VALUES (?, ?, ?, ?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    mission_id,
                    kind,
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )
    except Exception:
        # 不能讓 logging 拖垮主流程
        pass


def recent(limit: int = 100) -> list[dict[str, Any]]:
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT ts, mission_id, kind, payload FROM traces ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"ts": r[0], "mission_id": r[1], "kind": r[2], "payload": json.loads(r[3])}
            for r in rows
        ]
    except Exception:
        return []
