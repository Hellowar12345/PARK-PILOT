"""
FastAPI 後端
啟動：  uv run uvicorn demo.app:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import asyncio
from demo.inference import Predictor, SimulationSource, LiveSource
from demo.tdx_poller import run_poller

# PARK-PILOT agentic 層
from agent import agent_router, watcher_loop

STATIC_DIR = Path(__file__).parent / "static"
ROOT       = Path(__file__).parent.parent
RECENT_OBS = str(ROOT / "recent_obs.parquet")

# ── 全域 state（啟動時載入一次）────────────────────────────────────────
_predictor: Optional[Predictor] = None
_sim_source: Optional[SimulationSource] = None
_live_source: LiveSource = LiveSource()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _sim_source
    print("初始化推論引擎...")
    _sim_source = SimulationSource(RECENT_OBS)
    _predictor  = Predictor()
    # 啟動 TDX 背景 poller（若有設環境變數才會真正執行）
    asyncio.create_task(run_poller(_live_source))
    # 啟動 PARK-PILOT Watcher（監看感測器、自動觸發 re-plan）+ 監督重啟
    watcher_holder = {"task": asyncio.create_task(watcher_loop())}

    async def _supervise():
        while True:
            await asyncio.sleep(30)
            t = watcher_holder["task"]
            if t.done():
                print("Watcher 已停止，重新啟動")
                watcher_holder["task"] = asyncio.create_task(watcher_loop())

    supervisor_task = asyncio.create_task(_supervise())
    # 暖機 agent session 的 sim_now_iso（cheap, 只讀 recent_obs 一個欄位）
    try:
        import polars as pl
        from agent.state import session as agent_session
        if not agent_session.sim_now_iso:
            obs = pl.read_parquet(RECENT_OBS, columns=["DataCollectTime"])
            agent_session.sim_now_iso = str(obs["DataCollectTime"].max())[:19]
            print(f"PARK-PILOT sim_now_iso = {agent_session.sim_now_iso}")
    except Exception as e:
        print(f"sim_now warmup skipped: {e}")
    print("Ready. → http://localhost:8000/pilot 進入 PARK-PILOT 代理人介面")
    try:
        yield
    finally:
        supervisor_task.cancel()
        watcher_holder["task"].cancel()


app = FastAPI(title="停車空位預測 Demo + PARK-PILOT", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(agent_router)


@app.get("/pilot")
def pilot():
    return FileResponse(str(STATIC_DIR / "pilot.html"))


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/baseline")
def baseline():
    """模型 vs 基準對照圖（錄影/報告用，免跳出瀏覽器）。"""
    return FileResponse(str(ROOT / "baseline_compare.png"))


@app.get("/api/segments")
def get_segments():
    """回傳全部路段 metadata，給前端畫 base markers"""
    lkp = _predictor.lookup.select(
        ["ParkingSegmentID", "segment_name", "lat", "lon", "TotalSpaces", "fare_price"]
    ).drop_nulls(subset=["lat", "lon"])
    return lkp.to_dicts()


@app.get("/api/sim_timestamps")
def sim_timestamps():
    """
    回傳模擬模式可選用的歷史時間戳：
    最近 30 天，每天 6 個代表時間點（7/9/12/15/17/19 時）
    """
    obs = pl.read_parquet(RECENT_OBS).select("DataCollectTime")
    max_ts = obs["DataCollectTime"].max()
    # 從最新觀測往前 30 天
    try:
        base = datetime.fromisoformat(max_ts[:19])
    except Exception:
        base = datetime(2026, 4, 21, 19, 0)
    slots = []
    for day_offset in range(0, 30):
        day = base - timedelta(days=day_offset)
        for h in [7, 9, 12, 15, 17, 19]:
            slots.append(day.replace(hour=h, minute=0, second=0).isoformat(timespec="minutes"))
    return sorted(set(slots), reverse=True)[:120]


class PredictRequest(BaseModel):
    lat: float
    lon: float
    radius_km: float = 0.5
    eta_minutes: int = 15
    mode: str = "simulation"          # "simulation" | "live"
    sim_now: Optional[str] = None     # ISO datetime string，模擬模式用


class PredictResponse(BaseModel):
    segment_id: str
    name: str
    lat: float
    lon: float
    TotalSpaces: int
    distance_m: int
    fare_price: Optional[float]
    current_occ: Optional[float]
    current_avail: Optional[int]
    predicted_occ: float
    predicted_avail: int
    actual_occ: Optional[float]
    actual_avail: Optional[int]
    score: float


@app.post("/api/predict", response_model=list[PredictResponse])
def predict(req: PredictRequest):
    if _predictor is None:
        raise HTTPException(503, "模型尚未載入")

    if req.radius_km <= 0 or req.radius_km > 5:
        raise HTTPException(400, "radius_km 須在 0~5 km 之間")
    if req.eta_minutes < 0 or req.eta_minutes > 120:
        raise HTTPException(400, "eta_minutes 須在 0~120 之間")

    if req.mode == "simulation":
        if req.sim_now:
            try:
                now_dt = datetime.fromisoformat(req.sim_now)
            except ValueError:
                raise HTTPException(400, "sim_now 格式錯誤，請使用 ISO 格式（如 2025-12-15T18:00）")
        else:
            # 預設取最新觀測時間往前推 30 分鐘
            obs = pl.read_parquet(RECENT_OBS).select("DataCollectTime")
            max_ts = obs["DataCollectTime"].max()
            now_dt = datetime.fromisoformat(max_ts[:19]) - timedelta(minutes=30)

        if now_dt < datetime(2023, 5, 1):
            raise HTTPException(400, "模擬時間不可早於 2023-05-01")

        source = _sim_source
    else:
        source = _live_source

    arrival_dt = now_dt + timedelta(minutes=req.eta_minutes)

    results = _predictor.predict_at(
        dest_lat=req.lat,
        dest_lon=req.lon,
        radius_km=req.radius_km,
        now_dt=now_dt,
        arrival_dt=arrival_dt,
        source=source,
    )

    if not results:
        return []

    # 只回前 20 個
    return results[:20]


@app.get("/api/live_status")
def live_status():
    """回傳 Live 模式 buffer 狀態（有多少路段有足夠觀測）"""
    ready = sum(
        1 for seg_id in _predictor._seg_static
        if _live_source.count(seg_id) >= 6
    )
    total = len(_predictor._seg_static)
    return {"ready_segments": ready, "total_segments": total, "ready": ready >= 10}
