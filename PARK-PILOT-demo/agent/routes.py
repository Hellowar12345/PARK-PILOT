"""FastAPI 路由：/agent/* — 給前端 Glass Brain UI 與 demo 控制使用。"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .broker import broker
from .orchestrator import run_session
from .simulator import display_sim, sensor_sim
from .state import session
from . import traces

router = APIRouter(prefix="/agent", tags=["agent"])
STATIC_DIR = Path(__file__).parent.parent / "demo" / "static"


# ── 狀態與健康檢查 ───────────────────────────────────────────────────────
@router.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "state": session.state,
        "mission": session.mission.to_dict() if session.mission else None,
        "sim_now_iso": session.sim_now_iso,
    }


@router.get("/state")
async def get_state() -> dict:
    return {
        "state": session.state,
        "mission": session.mission.to_dict() if session.mission else None,
        "sensors": sensor_sim.snapshot(),
        "display": display_sim.snapshot(),
        "preferences": session.preferences,
        "sim_now_iso": session.sim_now_iso,
    }


# ── PLANNING：使用者下指令 ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    text: str


# 單一事件最長等待秒數（Gemini 串流/工具執行任一步驟卡住超過此值即視為逾時）
EVENT_TIMEOUT_S = 45


@router.post("/chat")
async def chat(req: ChatRequest):
    """觸發 PLANNING session。回傳 SSE。"""
    text = (req.text or "").strip()
    if not text:
        return {"ok": False, "error": "請先輸入目標地點"}
    if len(text) > 500:
        text = text[:500]
    # 新任務開始：清掉舊狀態
    session.messages = []
    session.mission = None
    session.last_replan_iso = ""
    session.last_user_text = text
    await display_sim.update(
        oled_line1="思考中…", oled_line2=text[:14], oled_line3="",
        led_color="blue_breathe",
    )

    async def gen():
        agen = run_session("PLANNING", text)
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(agen.__anext__(), timeout=EVENT_TIMEOUT_S)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    session.state = "IDLE"
                    yield f"data: {json.dumps({'type':'error','text':'Gemini 回應逾時，請重試'}, ensure_ascii=False)}\n\n"
                    break
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','text': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            await agen.aclose()
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── 全域事件 SSE ─────────────────────────────────────────────────────────
@router.get("/events")
async def events():
    """常駐 SSE：thinking / tool_call / sensor_event / display / alert 全推。"""
    q = broker.subscribe()

    async def gen():
        try:
            init = {
                "type": "init",
                "state": session.state,
                "sensors": sensor_sim.snapshot(),
                "display": display_sim.snapshot(),
                "mission": session.mission.to_dict() if session.mission else None,
                "sim_now_iso": session.sim_now_iso,
            }
            yield f"data: {json.dumps(init, ensure_ascii=False)}\n\n"
            while True:
                ev = await q.get()
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            broker.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── 模擬感測器控制 (取代將來的 ESP32 MQTT) ────────────────────────────────
class SlotAction(BaseModel):
    slot_id: int


@router.post("/sim/slot/occupy")
async def sim_occupy(req: SlotAction) -> dict:
    await sensor_sim.set_slot(req.slot_id, "occupied", source="manual_sim")
    return {"ok": True, "snapshot": sensor_sim.snapshot()}


@router.post("/sim/slot/free")
async def sim_free(req: SlotAction) -> dict:
    await sensor_sim.set_slot(req.slot_id, "free", source="manual_sim")
    return {"ok": True, "snapshot": sensor_sim.snapshot()}


@router.post("/sim/slot/toggle")
async def sim_toggle(req: SlotAction) -> dict:
    slots = {s["slot_id"]: s for s in sensor_sim.snapshot_list()}
    cur = slots.get(req.slot_id)
    if not cur:
        return {"ok": False, "error": "unknown slot"}
    new = "free" if cur["state"] == "occupied" else "occupied"
    await sensor_sim.set_slot(req.slot_id, new, source="manual_sim")
    return {"ok": True, "snapshot": sensor_sim.snapshot()}


class TargetSet(BaseModel):
    slot_id: int


@router.post("/sim/target")
async def sim_target(req: TargetSet) -> dict:
    sensor_sim.target_slot = req.slot_id
    await broker.publish({"type": "target_changed", "target_slot": req.slot_id})
    return {"ok": True, "target_slot": sensor_sim.target_slot}


class SimTime(BaseModel):
    iso: str


@router.post("/sim/time")
async def sim_time(req: SimTime) -> dict:
    session.sim_now_iso = req.iso
    return {"ok": True, "sim_now_iso": session.sim_now_iso}


# ── 重置 / Traces ────────────────────────────────────────────────────────
@router.post("/reset")
async def reset() -> dict:
    session.reset()
    await sensor_sim.reset_all()
    await display_sim.reset()
    await broker.publish({"type": "reset"})
    return {"ok": True}


@router.get("/traces")
async def traces_recent(limit: int = 100) -> dict:
    return {"ok": True, "traces": traces.recent(limit=limit)}


# ── 多智能體賽局模擬 ──────────────────────────────────────────────────────
class GameRequest(BaseModel):
    lat: float = 25.014
    lon: float = 121.4631
    at_iso: Optional[str] = None
    n_drivers: Optional[int] = None
    radius_km: float = 1.2
    scarcity: float = 0.8
    reflexive_frac: float = 0.3
    patience: int = 2
    max_segments: int = 6
    cap_max: Optional[int] = 4


@router.post("/game/run")
async def game_run(req: GameRequest) -> dict:
    """
    跑「同一個準預測、多人競爭」的三世界對照：
    各自為政(羊群崩潰) vs PARK-PILOT 協調分流 vs 少數反身推理。
    pred_free 來自真實 LightGBM 預測。
    """
    from .game import run_game
    at = req.at_iso or session.sim_now_iso or "2026-04-21T19:00:00"
    return run_game(
        dest_lat=req.lat, dest_lon=req.lon, at_iso=at,
        n_drivers=req.n_drivers, radius_km=req.radius_km,
        scarcity=req.scarcity, reflexive_frac=req.reflexive_frac,
        patience=req.patience, max_segments=req.max_segments,
        cap_max=req.cap_max,
    )


@router.get("/game")
async def game_page():
    return FileResponse(str(STATIC_DIR / "game.html"))


# ── 靜態頁 ───────────────────────────────────────────────────────────────
@router.get("/")
async def pilot_index():
    return FileResponse(str(STATIC_DIR / "pilot.html"))
