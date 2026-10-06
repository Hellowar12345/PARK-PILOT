"""
推論核心模組。
SimulationSource 從 recent_obs.parquet 模擬「現在」的即時觀測。
LiveSource 從記憶體 ring buffer 讀取（由 tdx_poller 填充）。
"""
from __future__ import annotations
import math
import pickle
import numpy as np
import polars as pl
import holidays as hd
from datetime import datetime, timedelta
from collections import deque
from typing import Optional

# ── 路徑常數（相對於專案根目錄，可攜到任何電腦）──────────────────────────
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH       = os.path.join(ROOT, "model_lgbm.pkl")
LOOKUP_PATH      = os.path.join(ROOT, "inference_lookup.parquet")
HIST_HDOW_PATH   = os.path.join(ROOT, "hist_hdow.parquet")
HIST_MONTH_PATH  = os.path.join(ROOT, "hist_month.parquet")
HIST_HOL_PATH    = os.path.join(ROOT, "hist_holiday.parquet")
RECENT_OBS_PATH  = os.path.join(ROOT, "recent_obs.parquet")

# 必須和 train.py FEATURE_COLS 完全一致（順序也要一樣）
_FEATURE_COLS_29 = [
    "hour", "month", "day_of_week", "is_holiday", "is_rush_hour",
    "temperature", "humidity",
    "hist_occ_hour_dow", "hist_occ_std", "hist_full_rate",
    "hist_occ_month", "hist_occ_holiday",
    "fare_price", "fare_start_hour", "fare_end_hour",
    "lat", "lon", "TotalSpaces", "dist_to_mrt",
    "lag_occ_1", "lag_occ_2", "lag_occ_3", "lag_occ_6", "lag_occ_12",
    "lag_occ_62",
    "roll_mean_3", "roll_std_3", "roll_mean_6",
    "target_enc_segment",
]
_FEATURE_COLS_32 = [
    "hour", "month", "day_of_week", "is_holiday", "is_rush_hour",
    "temperature", "humidity",
    "hist_occ_hour_dow", "hist_occ_std", "hist_full_rate",
    "hist_occ_month", "hist_occ_holiday",
    "fare_price", "fare_start_hour", "fare_end_hour",
    "lat", "lon", "TotalSpaces", "dist_to_mrt",
    "lag_occ_1", "lag_occ_2", "lag_occ_3", "lag_occ_6", "lag_occ_12",
    "lag_occ_62", "lag_occ_434",
    "roll_mean_3", "roll_std_3", "roll_mean_6", "roll_mean_12", "roll_std_6",
    "target_enc_segment",
]
# 啟動時由 Predictor 依 model.num_feature() 自動選擇
FEATURE_COLS: list[str] = _FEATURE_COLS_32

TW_HOLIDAYS = set(hd.country_holidays("TW", years=range(2023, 2027)).keys())


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(max(0.0, a)))


def _time_features(dt: datetime) -> dict:
    dow = dt.weekday()  # 0=Mon, 6=Sun
    hour = dt.hour
    month = dt.month
    is_hol = int(dt.strftime("%Y-%m-%d") in TW_HOLIDAYS)
    is_rush = int((7 <= hour <= 9) or (17 <= hour <= 19))
    return {
        "hour": hour,
        "month": month,
        "day_of_week": dow,
        "is_holiday": is_hol,
        "is_rush_hour": is_rush,
    }


def _build_lag_rolling(recent_occ: list[float]) -> dict:
    """
    recent_occ: list of recent occupancy_rate values, newest-last.
    最多需要 434 筆；lag_N = recent_occ[-N] if len >= N else nan
    """
    n = len(recent_occ)

    def lag(k):
        return float(recent_occ[-k]) if n >= k else float("nan")

    lags = [lag(1), lag(2), lag(3), lag(6), lag(12), lag(62), lag(434)]

    # rolling on the most recent 12 values (already shifted = no leakage)
    window6 = [v for v in recent_occ[-6:] if not math.isnan(v)]
    window3 = [v for v in recent_occ[-3:] if not math.isnan(v)]
    window12 = [v for v in recent_occ[-12:] if not math.isnan(v)]

    def safe_mean(lst): return float(np.mean(lst)) if len(lst) >= 2 else float("nan")
    def safe_std(lst):  return float(np.std(lst))  if len(lst) >= 2 else float("nan")

    return {
        "lag_occ_1":   lags[0],
        "lag_occ_2":   lags[1],
        "lag_occ_3":   lags[2],
        "lag_occ_6":   lags[3],
        "lag_occ_12":  lags[4],
        "lag_occ_62":  lags[5],
        "lag_occ_434": lags[6],
        "roll_mean_3":  safe_mean(window3),
        "roll_std_3":   safe_std(window3),
        "roll_mean_6":  safe_mean(window6),
        "roll_mean_12": safe_mean(window12),
        "roll_std_6":   safe_std(window6),
    }


# ── Data Source 抽象 ─────────────────────────────────────────────────────

class SimulationSource:
    """從 recent_obs.parquet 依模擬時間點抽取觀測序列"""

    def __init__(self, recent_obs_path: str = RECENT_OBS_PATH):
        print("  載入 recent_obs ...")
        self._obs = pl.read_parquet(recent_obs_path)
        # index by segment for fast lookup
        self._by_seg: dict[str, pl.DataFrame] = {}
        for seg_id, grp in self._obs.group_by("ParkingSegmentID"):
            self._by_seg[seg_id[0]] = grp.sort("DataCollectTime")

    def get_recent(self, segment_id: str, before_dt: datetime) -> list[float]:
        """回傳 before_dt 之前的 occupancy_rate 序列（新→舊，最新的在尾端）"""
        grp = self._by_seg.get(segment_id)
        if grp is None:
            return []
        cutoff = before_dt.isoformat(timespec="seconds").replace("+00:00", "")
        # 支援 ISO with timezone
        vals = (
            grp.filter(pl.col("DataCollectTime").str.slice(0, 19) < cutoff[:19])
            .tail(500)
            ["occupancy_rate"]
            .to_list()
        )
        return [v for v in vals if v is not None]

    def get_weather(self, segment_id: str, before_dt: datetime) -> dict:
        """取最近一筆天氣"""
        grp = self._by_seg.get(segment_id)
        if grp is None:
            return {"temperature": 25.0, "humidity": 70.0}
        cutoff = before_dt.isoformat(timespec="seconds")[:19]
        row = (
            grp.filter(pl.col("DataCollectTime").str.slice(0, 19) < cutoff)
            .tail(1)
        )
        if len(row) == 0:
            return {"temperature": 25.0, "humidity": 70.0}
        return {
            "temperature": float(row["temperature"][0] or 25.0),
            "humidity":    float(row["humidity"][0] or 70.0),
        }

    def get_current_obs(self, segment_id: str, before_dt: datetime):
        """取 before_dt 前最新一筆 (occupancy_rate, available)"""
        grp = self._by_seg.get(segment_id)
        if grp is None:
            return None, None
        cutoff = before_dt.isoformat(timespec="seconds")[:19]
        row = (
            grp.filter(pl.col("DataCollectTime").str.slice(0, 19) < cutoff)
            .tail(1)
        )
        if len(row) == 0:
            return None, None
        occ   = row["occupancy_rate"][0]
        avail = row["AvailableSpaces"][0]
        return occ, avail

    def get_actual_obs(self, segment_id: str, at_dt: datetime):
        """取 at_dt 之後最近一筆（模擬模式的 ground truth）"""
        grp = self._by_seg.get(segment_id)
        if grp is None:
            return None, None
        target_str = at_dt.isoformat(timespec="seconds")[:19]
        row = (
            grp.filter(pl.col("DataCollectTime").str.slice(0, 19) >= target_str)
            .head(1)
        )
        if len(row) == 0:
            return None, None
        occ   = row["occupancy_rate"][0]
        avail = row["AvailableSpaces"][0]
        return occ, avail


class LiveSource:
    """從記憶體 ring buffer 讀取（由 tdx_poller 填充）"""

    def __init__(self):
        self._buffer: dict[str, deque] = {}  # segment_id -> deque of (dt_str, occ, avail, temp, hum)

    def push(self, segment_id: str, dt_str: str, occ: float,
             avail: int, temperature: float = 25.0, humidity: float = 70.0):
        if segment_id not in self._buffer:
            self._buffer[segment_id] = deque(maxlen=500)
        self._buffer[segment_id].append((dt_str, occ, avail, temperature, humidity))

    def count(self, segment_id: str) -> int:
        return len(self._buffer.get(segment_id, []))

    def get_recent(self, segment_id: str, before_dt: datetime) -> list[float]:
        buf = self._buffer.get(segment_id, deque())
        cutoff = before_dt.isoformat(timespec="seconds")[:19]
        return [occ for (dt_str, occ, *_) in buf if dt_str[:19] < cutoff]

    def get_weather(self, segment_id: str, before_dt: datetime) -> dict:
        buf = self._buffer.get(segment_id, deque())
        for (_, _, _, temp, hum) in reversed(list(buf)):
            return {"temperature": temp or 25.0, "humidity": hum or 70.0}
        return {"temperature": 25.0, "humidity": 70.0}

    def get_current_obs(self, segment_id: str, before_dt: datetime):
        buf = self._buffer.get(segment_id, deque())
        cutoff = before_dt.isoformat(timespec="seconds")[:19]
        for (dt_str, occ, avail, *_) in reversed(list(buf)):
            if dt_str[:19] < cutoff:
                return occ, avail
        return None, None

    def get_actual_obs(self, segment_id, at_dt):
        return None, None  # live 模式無 ground truth


# ── 主 Predictor ──────────────────────────────────────────────────────────

class Predictor:
    def __init__(self):
        print("載入模型與 lookup ...")
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        self.lookup   = pl.read_parquet(LOOKUP_PATH)
        self.hist_hdow = pl.read_parquet(HIST_HDOW_PATH)
        self.hist_month = pl.read_parquet(HIST_MONTH_PATH)
        self.hist_hol  = pl.read_parquet(HIST_HOL_PATH)
        # 轉成 dict 加速 lookup
        self._seg_static: dict[str, dict] = {
            row["ParkingSegmentID"]: row
            for row in self.lookup.to_dicts()
        }
        self._hdow_idx: dict[tuple, float] = {}
        self._hdow_std_idx: dict[tuple, float] = {}
        self._hdow_full_idx: dict[tuple, float] = {}
        for row in self.hist_hdow.to_dicts():
            k = (row["ParkingSegmentID"], row["hour"], row["day_of_week"])
            self._hdow_idx[k]      = row.get("hist_occ_hour_dow")
            self._hdow_std_idx[k]  = row.get("hist_occ_std")
            self._hdow_full_idx[k] = row.get("hist_full_rate")
        self._month_idx: dict[tuple, float] = {
            (r["ParkingSegmentID"], r["month"]): r.get("hist_occ_month")
            for r in self.hist_month.to_dicts()
        }
        self._hol_idx: dict[tuple, Optional[float]] = {
            (r["ParkingSegmentID"], r["is_holiday"]): r.get("hist_occ_holiday")
            for r in self.hist_hol.to_dicts()
        }
        # 自動對齊特徵數量
        global FEATURE_COLS
        nf = self.model.num_feature()
        if nf == 29:
            FEATURE_COLS = _FEATURE_COLS_29
        elif nf == 32:
            FEATURE_COLS = _FEATURE_COLS_32
        else:
            raise RuntimeError(f"未預期的模型特徵數: {nf}")
        print(f"  {len(self._seg_static)} 路段 ready  (model features={nf})")

    def candidates_near(self, lat: float, lon: float, radius_km: float) -> list[str]:
        result = []
        for seg_id, s in self._seg_static.items():
            slat = s.get("lat")
            slon = s.get("lon")
            if slat is None or slon is None:
                continue
            if haversine_km(lat, lon, slat, slon) <= radius_km:
                result.append(seg_id)
        return result

    def _build_feature_row(self, seg_id: str, arrival_dt: datetime,
                           recent_occ: list[float], weather: dict) -> dict:
        s = self._seg_static.get(seg_id, {})
        tf = _time_features(arrival_dt)
        hour = tf["hour"]
        dow  = tf["day_of_week"]
        month = tf["month"]
        is_hol = tf["is_holiday"]

        k_hdow = (seg_id, hour, dow)
        row = {
            **tf,
            "temperature":       weather.get("temperature", 25.0),
            "humidity":          weather.get("humidity", 70.0),
            "hist_occ_hour_dow": self._hdow_idx.get(k_hdow),
            "hist_occ_std":      self._hdow_std_idx.get(k_hdow),
            "hist_full_rate":    self._hdow_full_idx.get(k_hdow),
            "hist_occ_month":    self._month_idx.get((seg_id, month)),
            "hist_occ_holiday":  self._hol_idx.get((seg_id, is_hol)),
            "fare_price":        s.get("fare_price"),
            "fare_start_hour":   s.get("fare_start_hour", 0),
            "fare_end_hour":     s.get("fare_end_hour", 24),
            "lat":               s.get("lat"),
            "lon":               s.get("lon"),
            "TotalSpaces":       s.get("TotalSpaces"),
            "dist_to_mrt":       s.get("dist_to_mrt"),
            "target_enc_segment": s.get("target_enc_segment"),
            **_build_lag_rolling(recent_occ),
        }
        return row

    def predict_at(
        self,
        dest_lat: float,
        dest_lon: float,
        radius_km: float,
        now_dt: datetime,
        arrival_dt: datetime,
        source,
    ) -> list[dict]:
        seg_ids = self.candidates_near(dest_lat, dest_lon, radius_km)
        if not seg_ids:
            return []

        rows = []
        feature_matrix = []
        for seg_id in seg_ids:
            recent_occ = source.get_recent(seg_id, now_dt)
            weather    = source.get_weather(seg_id, now_dt)
            feat_row   = self._build_feature_row(seg_id, arrival_dt, recent_occ, weather)
            feature_matrix.append([
                float(feat_row.get(c) or 0.0) if feat_row.get(c) is not None else float("nan")
                for c in FEATURE_COLS
            ])

            s = self._seg_static.get(seg_id, {})
            slat = s.get("lat", dest_lat)
            slon = s.get("lon", dest_lon)
            dist_m = haversine_km(dest_lat, dest_lon, slat, slon) * 1000
            cur_occ, cur_avail = source.get_current_obs(seg_id, now_dt)
            actual_occ, actual_avail = source.get_actual_obs(seg_id, arrival_dt)

            rows.append({
                "segment_id":   seg_id,
                "name":         s.get("segment_name", seg_id),
                "lat":          slat,
                "lon":          slon,
                "TotalSpaces":  int(s.get("TotalSpaces") or 0),
                "distance_m":   round(dist_m),
                "fare_price":   s.get("fare_price"),
                "current_occ":  cur_occ,
                "current_avail": cur_avail,
                "actual_occ":   actual_occ,
                "actual_avail": actual_avail,
            })

        X = np.array(feature_matrix, dtype=np.float32)
        preds = np.clip(self.model.predict(X), 0.0, 1.0)

        results = []
        for row, pred_occ in zip(rows, preds):
            total = row["TotalSpaces"] or 1
            pred_avail = max(0, round(total * (1.0 - float(pred_occ))))
            score = pred_avail / max(1.0, row["distance_m"] / 100.0)
            results.append({
                **row,
                "predicted_occ":   round(float(pred_occ), 4),
                "predicted_avail": pred_avail,
                "score":           round(score, 4),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
