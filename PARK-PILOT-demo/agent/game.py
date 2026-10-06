"""
把多智能體賽局接上真實路段 + LightGBM 預測。

pred_free（大家看到的公開訊號）來自你訓練的 LightGBM —— 不是捏造的數字。
為了讓「稀缺」浮現（歷史資料供給太足），用 scarcity 參數縮放開放車位數，
模擬尖峰/活動時的真實競爭。這是誠實的：我們明說「為展示競爭，把可用車位縮放成 X」。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from .world import GameSegment, run_selfish, run_coordinated, run_minority
from . import tools as T


def build_scenario(dest_lat: float, dest_lon: float, at_iso: str,
                   radius_km: float = 1.0, max_segments: int = 6,
                   scarcity: float = 0.5, cap_max: Optional[int] = None,
                   exclude_lots: bool = True) -> list[GameSegment]:
    """
    用 LightGBM 對目的地附近路段預測佔用率 → 換算 pred_free（公開訊號）。
    scarcity：把預測剩餘車位縮放成這個比例當「本場開放車位」，製造尖峰競爭。
    cap_max：每條街開放車位上限（模擬「路邊小車格」，讓羊群一湧就爆 → 戲劇性最強）。
    exclude_lots：濾掉名稱含「停車場」的大型場（本專題是路邊停車）。
    """
    try:
        arrival = T._parse_iso(at_iso)
    except Exception:
        arrival = datetime.now()

    # 多取一些候選，濾掉大型停車場後仍夠數
    cand = T.tool_list_candidate_segments(dest_lat, dest_lon, radius_km,
                                          top_k=max_segments * 3)
    chosen = []
    for c in cand.get("segments", []):
        if len(chosen) >= max_segments:
            break
        name = c.get("name") or c["seg_id"]
        if exclude_lots and "停車場" in name:
            continue
        chosen.append(c)

    # 一次批次呼叫 LightGBM（與 predict_occupancy 共用同一推論邏輯）
    preds = T.predict_occ_batch([c["seg_id"] for c in chosen], arrival)

    segs: list[GameSegment] = []
    for c in chosen:
        if c["seg_id"] not in preds:
            continue
        pred_occ, _ = preds[c["seg_id"]]
        total = int(c.get("total_spaces") or 0)
        pred_free_real = round(total * (1.0 - pred_occ))
        # 縮放製造稀缺（至少留 1，避免全 0）
        cap = max(1, round(pred_free_real * scarcity))
        if cap_max is not None:
            cap = min(cap, cap_max)   # 路邊小車格上限 → 羊群一湧就爆
        segs.append(GameSegment(
            seg_id=c["seg_id"],
            name=c.get("name") or c["seg_id"],
            capacity=cap,
            dist_m=int(c.get("dist_m") or 0),
            pred_free=cap,   # 公開訊號 = 本場開放車位（大家看到的「預測還有幾位」）
        ))
    return segs


def run_game(dest_lat: float, dest_lon: float, at_iso: str,
             n_drivers: Optional[int] = None, radius_km: float = 1.2,
             scarcity: float = 0.8, reflexive_frac: float = 0.3,
             patience: Optional[int] = None, max_segments: int = 6,
             cap_max: Optional[int] = 4, exclude_lots: bool = True) -> dict:
    """建場景 → 跑三個世界對照 → 回傳結構化結果（含殺手數字）。"""
    segs = build_scenario(dest_lat, dest_lon, at_iso, radius_km,
                          max_segments=max_segments, scarcity=scarcity,
                          cap_max=cap_max, exclude_lots=exclude_lots)
    if len(segs) < 2:
        return {"ok": False, "error": "附近可競爭路段不足 2 條，換目的地或放大半徑"}

    total_cap = sum(s.capacity for s in segs)
    if n_drivers is None:
        # 預設略微超量（車比位多一點）→ 各自為政會有人放棄，協調仍能塞下
        n_drivers = max(4, int(round(total_cap * 1.05)))

    # 三世界用同一個耐心上限 → 公平對照；世界 B 協調一次到位（本就不繞圈）
    a = run_selfish(segs, n_drivers, max_tries=patience)
    b = run_coordinated(segs, n_drivers)
    c = run_minority(segs, n_drivers, reflexive_frac=reflexive_frac, max_tries=patience)

    # 找「全空卻被浪費」的街（羊群悲劇的視覺重點）
    wasted = [x for x in a["seg_final"] if x["occupied"] == 0 and x["capacity"] > 0]

    return {
        "ok": True,
        "scenario": {
            "n_drivers": n_drivers, "total_capacity": total_cap,
            "scarcity": scarcity, "patience": patience,
            "segments": [{"seg_id": s.seg_id, "name": s.name,
                          "capacity": s.capacity, "dist_m": s.dist_m,
                          "pred_free": s.pred_free} for s in segs],
        },
        "world_a_selfish": a,
        "world_b_coordinated": b,
        "world_c_minority": c,
        "headline": {
            "selfish_gave_up": a["missed"],
            "selfish_reroutes": a["total_reroutes"],
            "coordinated_gave_up": b["missed"],
            "coordinated_reroutes": b["total_reroutes"],
            "wasted_empty_streets": [{"name": w["name"], "free": w["capacity"]} for w in wasted],
            "minority_vs_naive": c.get("by_group", {}),
        },
    }
