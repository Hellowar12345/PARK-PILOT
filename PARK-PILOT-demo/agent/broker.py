"""
非常輕量的 in-process pub/sub event broker。
所有「會被攝影機拍到」的事件（thinking、tool_call、感測器變化、display 更新、alert）
都丟進 broker；前端 /agent/events SSE 訂閱，Watcher 也訂閱。
"""
from __future__ import annotations
import asyncio
from typing import Any


class Broker:
    def __init__(self) -> None:
        self.subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self.subscribers:
            self.subscribers.remove(q)

    async def publish(self, event: dict[str, Any]) -> None:
        # 丟給所有訂閱者；queue 滿時直接 drop（避免阻塞）
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass


broker = Broker()
