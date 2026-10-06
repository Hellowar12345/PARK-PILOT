"""
SensorSim / DisplaySim — 把「實體硬體」抽象成純記憶體狀態 + 事件廣播。
之後接 MQTT 真實 ESP32 / Pi 鏡頭時，只要換掉這兩個類別的 publish 路徑，
agent 層、tool 層、UI 都不用動。
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .broker import broker


NUM_SLOTS = 6
DEFAULT_TARGET = 3  # slot_3 是 agent 默認監看的「目標格」


@dataclass
class SlotState:
    slot_id: int
    state: str = "free"          # free | occupied
    last_change_iso: str = ""
    confidence: float = 1.0
    source: str = "fused"        # fused | cam | ultra | manual_sim


class SensorSim:
    """模擬 Pi 鏡頭 + ESP32 超音波 的融合輸出。"""

    def __init__(self) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        self.slots: dict[int, SlotState] = {
            i: SlotState(slot_id=i, last_change_iso=now) for i in range(1, NUM_SLOTS + 1)
        }
        self.target_slot: int = DEFAULT_TARGET

    def snapshot_list(self) -> list[dict[str, Any]]:
        return [
            {
                "slot_id": s.slot_id,
                "state": s.state,
                "last_change": s.last_change_iso,
                "confidence": s.confidence,
                "source": s.source,
                "is_target": s.slot_id == self.target_slot,
            }
            for s in sorted(self.slots.values(), key=lambda x: x.slot_id)
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "target_slot": self.target_slot,
            "slots": self.snapshot_list(),
        }

    async def set_slot(self, slot_id: int, state: str, source: str = "manual_sim",
                       confidence: float = 1.0) -> None:
        if slot_id not in self.slots:
            raise ValueError(f"unknown slot {slot_id}")
        prev = self.slots[slot_id].state
        if prev == state:
            return
        now = datetime.now().isoformat(timespec="seconds")
        self.slots[slot_id].state = state
        self.slots[slot_id].last_change_iso = now
        self.slots[slot_id].source = source
        self.slots[slot_id].confidence = confidence
        await broker.publish({
            "type": "sensor_event",
            "slot_id": slot_id,
            "prev": prev,
            "next": state,
            "confidence": confidence,
            "source": source,
            "ts": now,
            "is_target": slot_id == self.target_slot,
        })

    async def reset_all(self) -> None:
        for i in self.slots:
            await self.set_slot(i, "free", source="reset")


sensor_sim = SensorSim()


class DisplaySim:
    """模擬 ESP32 OLED + WS2812 LED + 蜂鳴器 + TTS + LINE Notify。"""

    def __init__(self) -> None:
        self.oled_line1: str = "PARK-PILOT 待命中"
        self.oled_line2: str = "請對麥克風說目標"
        self.oled_line3: str = ""
        self.led_color: str = "off"   # off | blue_breathe | green_solid | red_flash | yellow
        self.last_alert: str = ""
        self.last_tts: str = ""
        self.last_line: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "oled_line1": self.oled_line1,
            "oled_line2": self.oled_line2,
            "oled_line3": self.oled_line3,
            "led_color": self.led_color,
            "last_alert": self.last_alert,
            "last_tts": self.last_tts,
            "last_line": self.last_line,
        }

    async def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        await broker.publish({"type": "display_update", **self.snapshot()})

    async def alert(self, severity: str, reason: str, tts: str = "", line: str = "") -> None:
        self.last_alert = reason
        self.last_tts = tts
        self.last_line = line
        if severity == "alert":
            self.led_color = "red_flash"
        await broker.publish({
            "type": "alert",
            "severity": severity,
            "reason": reason,
            "tts": tts,
            "line": line,
            **self.snapshot(),
        })

    async def reset(self) -> None:
        self.oled_line1 = "PARK-PILOT 待命中"
        self.oled_line2 = "請對麥克風說目標"
        self.oled_line3 = ""
        self.led_color = "off"
        self.last_alert = ""
        self.last_tts = ""
        self.last_line = ""
        await broker.publish({"type": "display_update", **self.snapshot()})


display_sim = DisplaySim()
