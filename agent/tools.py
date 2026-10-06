"""
PARK-PILOT 10 個 tool 的實作。

設計鐵則：
* 工具的「外部世界」(感測器、顯示、TDX、LightGBM、天氣) 全部走介面層，將來換真實硬體時不用動 agent 層。
* predict_occupancy 直接重用 demo/inference.py 的 Predictor（既有 LightGBM 模型）。
"""
from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl

# 讓 from demo.inference import ... 可以動 (Predictor 內部使用絕對路徑載入 parquet，安全)
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo import inference  # noqa: E402  — 動態取 FEATURE_COLS
from demo.inference import Predictor, SimulationSource  # noqa: E402

from .simulator import sensor_sim, display_sim  # noqa: E402
from .state import session, ActiveMission  # noqa: E402
from .broker import broker  # noqa: E402
from . import traces  # noqa: E402
from .risk import (get_risk_table, expected_regret, value_of_information,  # noqa: E402
                   MISS_PENALTY, WALK_PENALTY_PER_100M)


# ── Lazy singletons (Predictor 載入要幾秒) ─────────────────────────────────
_predictor: Optional[Predictor] = None
_sim_source: Optional[SimulationSource] = None


def get_predictor() -> tuple[Predictor, SimulationSource]:
    global _predictor, _sim_source
    if _predictor is None:
        _predictor = Predictor()
        _sim_source = SimulationSource()
        # 自動推一個合理的 sim_now：用 recent_obs 中最大時間 - 30 分
        if not session.sim_now_iso:
            try:
                obs = pl.read_parquet(str(ROOT / "recent_obs.parquet")).select("DataCollectTime")
                max_ts = obs["DataCollectTime"].max()
                # 切掉時區 +08:00
                session.sim_now_iso = max_ts[:19]
            except Exception:
                session.sim_now_iso = "2026-04-21T17:30:00"
    return _predictor, _sim_source  # type: ignore[return-value]


# ── 工具 1：geocode_destination ───────────────────────────────────────────
LANDMARKS: dict[str, tuple[float, float]] = {
    "板橋車站":     (25.0140, 121.4631),
    "板橋火車站":   (25.0140, 121.4631),
    "新埔捷運站":   (25.0359, 121.4685),
    "新埔":         (25.0359, 121.4685),
    "府中捷運站":   (25.0095, 121.4595),
    "府中":         (25.0095, 121.4595),
    "江子翠捷運站": (25.0303, 121.4761),
    "江子翠":       (25.0303, 121.4761),
    "中和":         (24.9988, 121.4990),
    "永和":         (25.0098, 121.5168),
    "三重":         (25.0613, 121.4858),
    "林口長庚":     (25.0764, 121.3678),
    "淡水老街":     (25.1722, 121.4400),
    "西門町":       (25.0422, 121.5066),
    "台北101":      (25.0337, 121.5645),
    "台北車站":     (25.0478, 121.5170),
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(max(0.0, a)))


def tool_geocode_destination(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text in LANDMARKS:
        lat, lon = LANDMARKS[text]
        return {"ok": True, "lat": lat, "lon": lon, "name": text, "source": "local_dict"}
    for k, (lat, lon) in LANDMARKS.items():
        if k in text or text in k:
            return {"ok": True, "lat": lat, "lon": lon, "name": k, "source": "local_dict_fuzzy"}
    return {"ok": False, "error": f"無法解析地名: {text}（已知地標: {list(LANDMARKS.keys())[:6]} 等）"}


# ── 工具 2：get_weather_at_arrival ────────────────────────────────────────
def tool_get_weather_at_arrival(lat: float, lon: float, at_iso: str) -> dict[str, Any]:
    try:
        _, src = get_predictor()
        # 從 recent_obs 取任一路段的最近天氣作為代理
        target = _parse_iso(at_iso)
        for seg_id in list(src._by_seg.keys())[:5]:
            w = src.get_weather(seg_id, target)
            if w:
                return {
                    "ok": True,
                    "temperature": round(w.get("temperature", 25.0), 1),
                    "humidity": round(w.get("humidity", 70.0), 1),
                    "precip_mm": 0.0,
                    "condition": "雨" if (w.get("humidity") or 70) > 90 else "多雲",
                    "source": "sim/recent_obs",
                }
    except Exception:
        pass
    return {"ok": True, "temperature": 25.0, "humidity": 70.0,
            "precip_mm": 0.0, "condition": "default", "source": "fallback"}


# ── 工具 3：list_candidate_segments ───────────────────────────────────────
def tool_list_candidate_segments(lat: float, lon: float,
                                  radius_km: float = 0.5, top_k: int = 18) -> dict[str, Any]:
    predictor, _ = get_predictor()
    results: list[dict[str, Any]] = []
    for row in predictor.lookup.iter_rows(named=True):
        slat, slon = row.get("lat"), row.get("lon")
        if slat is None or slon is None:
            continue
        dist = _haversine_km(lat, lon, slat, slon)
        if dist <= radius_km:
            results.append({
                "seg_id": row["ParkingSegmentID"],
                "name": row.get("segment_name") or row["ParkingSegmentID"],
                "lat": slat,
                "lon": slon,
                "total_spaces": int(row.get("TotalSpaces") or 0),
                "fare": row.get("fare_price"),
                "dist_to_mrt_km": row.get("dist_to_mrt"),
                "dist_m": int(dist * 1000),
            })
    results.sort(key=lambda x: x["dist_m"])
    return {"ok": True, "radius_km": radius_km, "count": len(results[:top_k]),
            "segments": results[:top_k]}


# ── 工具 4：predict_occupancy (呼叫 LightGBM 預測模型) ─────────────────────
def _parse_iso(s: str) -> datetime:
    s = (s or "").strip()
    # 去掉時區字串簡化處理
    for tz in ("+08:00", "+0800", "Z"):
        if s.endswith(tz):
            s = s[: -len(tz)]
    return datetime.fromisoformat(s[:19] if len(s) >= 19 else s)


def predict_occ_batch(seg_ids: list[str], arrival_dt: datetime) -> dict[str, tuple[float, int]]:
    """
    對多個路段一次性呼叫 LightGBM（堆成單一矩陣批次推論，比逐列快）。
    回傳 {seg_id: (pred_occ, total_spaces)}；未知路段不在 dict 中。
    模擬模式：「現在」= arrival_dt，用 arrival 前的觀測算 lag（與訓練對齊）。
    predict_occupancy 與 game.build_scenario 共用此函式，避免特徵邏輯重複。
    """
    predictor, src = get_predictor()
    feature_cols = inference.FEATURE_COLS
    rows, valid = [], []
    for seg_id in seg_ids:
        s = predictor._seg_static.get(seg_id)
        if not s:
            continue
        recent = src.get_recent(seg_id, arrival_dt)
        weather = src.get_weather(seg_id, arrival_dt)
        feat = predictor._build_feature_row(seg_id, arrival_dt, recent, weather)
        rows.append([float(feat.get(c) or 0.0) if feat.get(c) is not None else float("nan")
                     for c in feature_cols])
        valid.append((seg_id, int(s.get("TotalSpaces") or 1)))
    if not rows:
        return {}
    preds = np.clip(predictor.model.predict(np.array(rows, dtype=np.float32)), 0.0, 1.0)
    return {seg_id: (float(p), total) for (seg_id, total), p in zip(valid, preds)}


def tool_predict_occupancy(segment_ids: list[str], at_iso: str) -> dict[str, Any]:
    predictor, src = get_predictor()
    feature_cols = inference.FEATURE_COLS
    try:
        arrival_dt = _parse_iso(at_iso)
    except Exception as e:
        return {"ok": False, "error": f"at_iso 格式錯誤: {at_iso} ({e})"}

    preds = predict_occ_batch(segment_ids, arrival_dt)
    if not preds:
        return {"ok": False,
                "error": f"所有 {len(segment_ids)} 個路段都不在資料庫，請改用 list_candidate_segments 取得有效路段 ID"}

    out: list[dict[str, Any]] = []
    for seg_id in segment_ids:
        if seg_id not in preds:
            out.append({"seg_id": seg_id, "error": "unknown segment"})
            continue
        pred, total = preds[seg_id]
        s = predictor._seg_static.get(seg_id, {})
        avail = max(0, round(total * (1.0 - pred)))

        # ── 風險：用 748 天歷史經驗分位數表達不確定性 ──
        ra = get_risk_table().assess(seg_id, arrival_dt, total, pred)
        out.append({
            "seg_id": seg_id,
            "name": s.get("segment_name") or seg_id,
            "pred_occ": round(pred, 4),
            "pred_avail": avail,
            "total_spaces": total,
            # 不確定性（經驗分位數，非校準機率）
            "p_available": round(ra.p_available, 3),
            "p_full": round(ra.p_full, 3),
            "avail_optimistic": ra.avail_p10,    # 樂觀(p10 佔用)
            "avail_likely": ra.avail_p50,        # 中位
            "avail_worst": ra.avail_p90,         # 悲觀(p90 佔用)
            "risk_confidence": ra.confidence,    # 樣本可信度
            "n_history": ra.n_samples,
        })
    return {
        "ok": True,
        "model": f"LightGBM ({len(feature_cols)} feats, MAE≈0.02) + 748天經驗分位數",
        "note": "p_available/avail_worst 為歷史經驗頻率(非校準機率)，用於表達風險而非保證",
        "at_iso": at_iso,
        "count": len(out),
        "predictions": out,
    }


# ── 工具：decide_with_regret（後悔值決策 + VoI）────────────────────────────
def tool_decide_with_regret(candidates: list[dict], rain: bool = False,
                            at_iso: Optional[str] = None) -> dict[str, Any]:
    """
    在已預測的候選中，用「期望後悔最小」而非「預測最空」挑選。
    candidates: list of {seg_id, name, dist_m, p_available, avail_likely, avail_worst}
                （通常直接餵 predict_occupancy 的 predictions + 各自距離）
    at_iso：抵達時間；應與 predict_occupancy 用的相同，使風險評估時間一致。
    """
    predictor, _ = get_predictor()
    table = get_risk_table()
    iso = at_iso or session.sim_now_iso
    try:
        arrival_dt = _parse_iso(iso) if iso else datetime.now()
    except Exception:
        arrival_dt = datetime.now()

    scored = []
    for c in candidates:
        seg_id = c.get("seg_id")
        s = predictor._seg_static.get(seg_id, {})
        total = int(s.get("TotalSpaces") or c.get("total_spaces") or 1)
        # 若呼叫端沒帶 p_available，就現算
        pred_occ = c.get("pred_occ")
        if pred_occ is None and c.get("avail_likely") is not None:
            pred_occ = 1.0 - (c["avail_likely"] / max(1, total))
        ra = table.assess(seg_id, arrival_dt, total, float(pred_occ or 0.5))
        dist_m = int(c.get("dist_m") or 0)
        rr = expected_regret(ra, dist_m, name=c.get("name", seg_id), rain=rain)
        voi = value_of_information(ra, detour_m=dist_m)
        d = rr.to_dict()
        d["voi"] = voi
        scored.append(d)

    scored.sort(key=lambda x: x["expected_regret"])
    # 同時給「貪心(只看預測最空)」的選擇，方便對照
    greedy = max(candidates, key=lambda c: c.get("avail_likely", 0)) if candidates else None
    return {
        "ok": True,
        "weights": {"miss_penalty": MISS_PENALTY, "walk_per_100m": WALK_PENALTY_PER_100M,
                    "rain": rain},
        "ranked_by_regret": scored,
        "best": scored[0] if scored else None,
        "greedy_choice": {"seg_id": greedy.get("seg_id"), "name": greedy.get("name")} if greedy else None,
        "note": "best = 期望後悔最小；greedy_choice = 只看預測最空(可能撲空)。兩者不同時最值得說明。",
    }


# ── 工具 5：get_live_tdx ──────────────────────────────────────────────────
def tool_get_live_tdx(seg_ids: list[str]) -> dict[str, Any]:
    _, src = get_predictor()
    try:
        sim_dt = _parse_iso(session.sim_now_iso) if session.sim_now_iso else datetime.now()
    except Exception:
        sim_dt = datetime.now()
    out: list[dict[str, Any]] = []
    for seg_id in seg_ids:
        occ, avail = src.get_current_obs(seg_id, sim_dt)
        out.append({
            "seg_id": seg_id,
            "occ_now": float(occ) if occ is not None else None,
            "avail_now": int(avail) if avail is not None else None,
            "source": "sim/recent_obs",
        })
    return {"ok": True, "count": len(out), "live": out}


# ── 工具 6：read_edge_sensor ──────────────────────────────────────────────
def tool_read_edge_sensor(slot_id: Optional[int] = None) -> dict[str, Any]:
    slots = sensor_sim.snapshot_list()
    if slot_id is not None:
        match = next((s for s in slots if s["slot_id"] == slot_id), None)
        return {"ok": match is not None,
                "target_slot": sensor_sim.target_slot, "slot": match}
    return {"ok": True, "target_slot": sensor_sim.target_slot, "slots": slots}


# ── 工具 7：get_route_eta ─────────────────────────────────────────────────
def tool_get_route_eta(from_lat: float, from_lon: float,
                       to_lat: float, to_lon: float) -> dict[str, Any]:
    dist_km = _haversine_km(from_lat, from_lon, to_lat, to_lon)
    eta_min = max(1, int(round(dist_km / 25 * 60)))   # 市區平均 25 km/h
    return {"ok": True, "distance_km": round(dist_km, 2),
            "eta_min": eta_min, "method": "haversine_x_25kmh"}


# ── 工具 8：lock_mission ──────────────────────────────────────────────────
_mission_counter = 0


async def tool_lock_mission(primary: str, primary_name: str, backups: list[str],
                             arrival_iso: str, strategy_note: str) -> dict[str, Any]:
    global _mission_counter
    _mission_counter += 1
    mid = "M" + datetime.now().strftime("%Y%m%d%H%M%S") + f"-{_mission_counter}"
    mission = ActiveMission(
        mission_id=mid,
        primary_seg=primary,
        primary_name=primary_name,
        backups=list(backups or []),
        strategy_note=strategy_note,
        arrival_iso=arrival_iso,
        locked_at_iso=datetime.now().isoformat(timespec="seconds"),
    )
    session.mission = mission

    arrival_hm = arrival_iso[11:16] if "T" in arrival_iso else arrival_iso
    backups_str = ",".join(backups[:2]) if backups else "—"
    await display_sim.update(
        oled_line1=f"目標 {primary_name[:12]}",
        oled_line2=f"抵達 {arrival_hm}",
        oled_line3=f"備案 {backups_str}",
        led_color="green_solid",
    )

    traces.log("mission_locked", mission.to_dict(), mission_id=mid)
    await broker.publish({"type": "mission_locked", "mission": mission.to_dict()})
    return {"ok": True, "mission_id": mid, "ack": True}


# ── 工具 9：alert_user ────────────────────────────────────────────────────
async def tool_alert_user(severity: str, reason: str,
                           tts: str, line: str) -> dict[str, Any]:
    await display_sim.alert(severity=severity, reason=reason, tts=tts, line=line)
    traces.log(
        "alert",
        {"severity": severity, "reason": reason, "tts": tts, "line": line},
        mission_id=session.mission.mission_id if session.mission else "",
    )
    return {"ok": True, "ack": True}


# ── Dispatcher ────────────────────────────────────────────────────────────
TOOL_FUNCTIONS = {
    "geocode_destination":     tool_geocode_destination,
    "get_weather_at_arrival":  tool_get_weather_at_arrival,
    "list_candidate_segments": tool_list_candidate_segments,
    "predict_occupancy":       tool_predict_occupancy,
    "get_live_tdx":            tool_get_live_tdx,
    "read_edge_sensor":        tool_read_edge_sensor,
    "get_route_eta":           tool_get_route_eta,
    "decide_with_regret":      tool_decide_with_regret,
    "lock_mission":            tool_lock_mission,
    "alert_user":              tool_alert_user,
}

ASYNC_TOOLS = {"lock_mission", "alert_user"}


async def dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"ok": False, "error": f"unknown tool {name}"}
    try:
        if name in ASYNC_TOOLS:
            return await fn(**(args or {}))  # type: ignore[misc]
        return fn(**(args or {}))            # type: ignore[misc]
    except TypeError as e:
        return {"ok": False, "error": f"參數錯誤: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"執行錯誤: {e}"}
