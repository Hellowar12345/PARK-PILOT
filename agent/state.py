"""SessionState — 全 agent 共用的狀態 (in-memory singleton)。"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class ActiveMission:
    mission_id: str
    primary_seg: str
    primary_name: str
    backups: list[str]
    strategy_note: str
    arrival_iso: str
    locked_at_iso: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "primary_seg": self.primary_seg,
            "primary_name": self.primary_name,
            "backups": list(self.backups),
            "strategy_note": self.strategy_note,
            "arrival_iso": self.arrival_iso,
            "locked_at_iso": self.locked_at_iso,
        }


@dataclass
class SessionState:
    state: str = "IDLE"                # IDLE | PLANNING | MONITORING | REPLANNING
    mission: Optional[ActiveMission] = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_user_text: str = ""
    last_replan_iso: str = ""
    sim_now_iso: str = ""              # 用於告訴 agent「現在是 sim 哪天幾點」
    preferences: dict[str, Any] = field(default_factory=dict)

    def can_replan(self, cooldown_s: int = 45) -> bool:
        if not self.last_replan_iso:
            return True
        try:
            last = datetime.fromisoformat(self.last_replan_iso)
        except Exception:
            return True
        return (datetime.now() - last).total_seconds() >= cooldown_s

    def reset(self) -> None:
        self.state = "IDLE"
        self.mission = None
        self.messages = []
        self.last_user_text = ""
        self.last_replan_iso = ""


session = SessionState()
