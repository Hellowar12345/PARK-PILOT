"""
Watcher 背景任務：
* 訂閱 broker 的 sensor_event
* 只在 MONITORING + 有 active mission + 目標格被佔 + cooldown 過了 時觸發 REPLAN
* 把 [EDGE EVENT] 摘要塞進新一輪對話，呼叫 orchestrator.run_session
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime

from .broker import broker
from .orchestrator import run_session
from .state import session
from . import traces

logger = logging.getLogger("watcher")
REPLAN_COOLDOWN_S = 45


async def watcher_loop() -> None:
    q = broker.subscribe()
    try:
        while True:
            ev = await q.get()
            if ev.get("type") != "sensor_event":
                continue
            if session.state != "MONITORING":
                continue
            if not session.mission:
                continue
            if not ev.get("is_target"):
                continue
            if ev.get("next") != "occupied":
                continue
            if not session.can_replan(REPLAN_COOLDOWN_S):
                await broker.publish({"type": "replan_skipped",
                                      "reason": "cooldown_45s_未過"})
                continue

            session.last_replan_iso = datetime.now().isoformat(timespec="seconds")
            await broker.publish({
                "type": "edge_event_detected",
                "slot": ev["slot_id"],
                "confidence": ev["confidence"],
                "source": ev.get("source", ""),
            })
            traces.log("watcher_trigger_replan", ev,
                       mission_id=session.mission.mission_id)

            m = session.mission
            edge_msg = (
                f"[EDGE EVENT] 目標 slot_{ev['slot_id']} 偵測到狀態變化: "
                f"free→{ev['next']}, 融合 confidence={ev['confidence']}, "
                f"source={ev.get('source','sensor_sim')}. "
                f"原任務主推路段 {m.primary_seg} ({m.primary_name}) 視為失效。"
                f"\n請進入 REPLANNING：\n"
                f"  1) 在 thinking 解釋為何 re-plan（提到 confidence 與目標格）\n"
                f"  2) 呼叫 predict_occupancy 評估備案 {m.backups} 在抵達時 ({m.arrival_iso}) 的佔用率\n"
                f"  3) 呼叫 lock_mission 換新主案（exclude={m.primary_seg}）\n"
                f"  4) 最後務必 alert_user(severity=\"alert\") 通知使用者，"
                f"tts/line 要明確說「目標格被搶走，改建議 XX 路段」"
            )

            # 非同步在背景跑 REPLAN，避免阻塞 watcher 訂閱
            asyncio.create_task(_run_replan(edge_msg))
    except asyncio.CancelledError:
        logger.info("Watcher cancelled")
        raise
    except Exception:
        # 單次迴圈出錯不可讓整個 watcher 死掉（否則監看功能永久失效）
        logger.exception("watcher loop error, continuing")
    finally:
        broker.unsubscribe(q)


REPLAN_TIMEOUT_S = 60


async def _run_replan(edge_msg: str) -> None:
    """背景跑 REPLANNING；失敗或逾時都把狀態拉回 MONITORING，避免卡死。"""
    agen = run_session("REPLANNING", edge_msg, prior_messages=session.messages)
    try:
        while True:
            try:
                await asyncio.wait_for(agen.__anext__(), timeout=REPLAN_TIMEOUT_S)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                logger.warning("REPLAN 逾時")
                await broker.publish({"type": "error", "text": "重新規劃逾時，已恢復監看"})
                break
    except Exception:
        logger.exception("REPLAN 失敗")
        await broker.publish({"type": "error", "text": "重新規劃失敗，已恢復監看"})
    finally:
        await agen.aclose()
        # 確保不會卡在 REPLANNING
        if session.mission:
            session.state = "MONITORING"
            await broker.publish({"type": "state_change", "state": "MONITORING"})
