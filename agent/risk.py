"""
風險與後悔值決策模組。

核心想法（這是專題「想法升級」的地基）：
* 點預測 pred_occ 只給期望值；真正的決策需要「至少有位的機率」與「最壞情況」。
* 我們用 748 天歷史的「經驗分位數」(risk_quantiles.parquet) 表達不確定性 ——
  這不是常態假設、也不宣稱是校準過的模型機率，而是歷史實際頻率（最誠實）。
* 決策準則從「argmax(預測最空)」升級成「argmin(期望後悔 regret)」：
  撲空(到了沒位)的痛 >> 多走幾步的痛，所以一個「分數高但 35% 機率撲空」的車位
  未必比「分數中等但 92% 有位」的車位好。
* 資訊價值(VoI)：當不確定性高、且去確認的成本(繞路)不划算時，agent 可以「主動選擇不去看 / 不賭」。

day_of_week 對齊：risk_quantiles.parquet 用 polars ISO (1=Mon..7=Sun)；
Python datetime.weekday() 是 0=Mon..6=Sun，所以查詢時 +1。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

RISK_PATH = str(Path(__file__).parent.parent / "risk_quantiles.parquet")

# 後悔值權重（可調；demo 時可當旋鈕）
MISS_PENALTY = 100.0     # 「到了卻沒位」的後悔（繞回頭、重找，痛）
WALK_PENALTY_PER_100M = 6.0   # 每多走 100m 的後悔
MIN_SAMPLES = 20         # 樣本太少的歷史格不可信


@dataclass
class RiskAssessment:
    seg_id: str
    p_available: float       # 抵達時至少 1 位的機率（歷史經驗）
    p_full: float            # 客滿機率
    occ_p10: float           # 樂觀（10 百分位佔用率）
    occ_p50: float           # 中位
    occ_p90: float           # 悲觀（90 百分位）
    avail_p10: int           # 樂觀空位數
    avail_p50: int
    avail_p90: int           # 悲觀空位數（最壞情況≈這麼少）
    n_samples: int
    confidence: str          # high | medium | low（依樣本數）
    source: str              # empirical | fallback

    def to_dict(self) -> dict:
        return {
            "seg_id": self.seg_id,
            "p_available": round(self.p_available, 3),
            "p_full": round(self.p_full, 3),
            "occ_p10": round(self.occ_p10, 3),
            "occ_p50": round(self.occ_p50, 3),
            "occ_p90": round(self.occ_p90, 3),
            "avail_p10": self.avail_p10,
            "avail_p50": self.avail_p50,
            "avail_p90": self.avail_p90,
            "n_samples": self.n_samples,
            "confidence": self.confidence,
            "source": self.source,
        }


def _cdf_at(x: float, p10: float, p50: float, p90: float) -> float:
    """
    用分位數帶 (occ 在 10/50/90 百分位) 建一條分段線性 CDF，
    回傳 P(occ <= x)。錨點 (0,0) 與 (1,1) 夾住兩端。
    用途：把「今天平移後的佔用率分布」轉成「至少有位的機率」，
    讓不確定性欄目跟著當下預測走，而非只用歷史頻率。
    """
    pts = [(0.0, 0.0), (p10, 0.10), (p50, 0.50), (p90, 0.90), (1.0, 1.0)]
    # 確保 x 單調（分位數已單調，clamp 後仍單調）
    if x <= pts[0][0]:
        return 0.0
    if x >= pts[-1][0]:
        return 1.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 - x0 < 1e-9:
                return y1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return 0.5


class RiskTable:
    def __init__(self, path: str = RISK_PATH):
        self._idx: dict[tuple, dict] = {}
        try:
            df = pl.read_parquet(path)
            for row in df.iter_rows(named=True):
                k = (row["ParkingSegmentID"], int(row["hour"]), int(row["day_of_week"]))
                self._idx[k] = row
            self.loaded = True
        except Exception:
            self.loaded = False

    def assess(self, seg_id: str, arrival_dt: datetime,
               total_spaces: int, pred_occ: float) -> RiskAssessment:
        """結合 LightGBM 點預測 + 歷史經驗分位數，回傳完整風險評估。"""
        dow_iso = arrival_dt.weekday() + 1   # 0..6 → 1..7
        row = self._idx.get((seg_id, arrival_dt.hour, dow_iso))
        total = max(1, int(total_spaces or 1))

        if row is None or (row.get("n") or 0) < MIN_SAMPLES:
            # fallback：沒有可信歷史，用點預測 ± 粗略不確定性帶
            occ50 = float(pred_occ)
            occ10 = max(0.0, occ50 - 0.15)
            occ90 = min(1.0, occ50 + 0.15)
            p_avail = 1.0 if occ90 < 0.999 else 0.5
            return RiskAssessment(
                seg_id=seg_id,
                p_available=p_avail, p_full=max(0.0, occ50 - 0.85) if occ50 > 0.85 else 0.05,
                occ_p10=occ10, occ_p50=occ50, occ_p90=occ90,
                avail_p10=round(total * (1 - occ10)),
                avail_p50=round(total * (1 - occ50)),
                avail_p90=round(total * (1 - occ90)),
                n_samples=int(row.get("n", 0)) if row else 0,
                confidence="low", source="fallback",
            )

        # 用 LightGBM 點預測「平移」歷史分位數帶：
        # 歷史中位 occ_p50 反映該時段典型水位；pred_occ 反映「今天」(含 lag/天氣)的調整。
        # shift = 今天比典型高/低多少，套用到整條分位數帶。
        hist_p50 = float(row["occ_p50"])
        shift = float(pred_occ) - hist_p50
        occ10 = min(1.0, max(0.0, float(row["occ_p10"]) + shift))
        occ50 = min(1.0, max(0.0, float(row["occ_p50"]) + shift))
        occ90 = min(1.0, max(0.0, float(row["occ_p90"]) + shift))

        n = int(row["n"])
        conf = "high" if n >= 100 else "medium" if n >= MIN_SAMPLES else "low"

        # p_available / p_full 跟著「今天平移後的分布」算，而非只用歷史頻率：
        # 一格客滿 = avail 取整為 0，即 occ >= 1 - 0.5/total。
        full_occ = 1.0 - 0.5 / total
        p_avail = _cdf_at(full_occ, occ10, occ50, occ90)   # P(occ < 客滿門檻) = 有位機率
        # 與歷史頻率取折衷，兼顧「當下預測」與「長期經驗」(各半)
        p_avail = 0.5 * p_avail + 0.5 * float(row["p_avail_ge1"])

        return RiskAssessment(
            seg_id=seg_id,
            p_available=p_avail,
            p_full=1.0 - p_avail,
            occ_p10=occ10, occ_p50=occ50, occ_p90=occ90,
            avail_p10=max(0, round(total * (1 - occ10))),   # 注意：occ 低→空位多
            avail_p50=max(0, round(total * (1 - occ50))),
            avail_p90=max(0, round(total * (1 - occ90))),
            n_samples=n, confidence=conf, source="empirical",
        )


@dataclass
class RegretResult:
    seg_id: str
    name: str
    expected_regret: float
    miss_term: float        # 撲空風險貢獻
    walk_term: float        # 步行成本貢獻
    p_available: float
    pred_avail: int
    avail_worst: int        # 悲觀情況空位（p90 occ）
    dist_m: int
    rationale: str

    def to_dict(self) -> dict:
        return {
            "seg_id": self.seg_id,
            "name": self.name,
            "expected_regret": round(self.expected_regret, 1),
            "miss_term": round(self.miss_term, 1),
            "walk_term": round(self.walk_term, 1),
            "p_available": round(self.p_available, 3),
            "pred_avail": self.pred_avail,
            "avail_worst": self.avail_worst,
            "dist_m": self.dist_m,
            "rationale": self.rationale,
        }


def expected_regret(ra: RiskAssessment, dist_m: int, name: str = "",
                    rain: bool = False) -> RegretResult:
    """
    期望後悔 = 撲空機率 × 撲空懲罰 + 步行成本。
    下雨時步行成本加重（雨天更不想走遠）。
    """
    p_miss = max(0.0, 1.0 - ra.p_available)
    miss_term = p_miss * MISS_PENALTY
    walk_w = WALK_PENALTY_PER_100M * (1.6 if rain else 1.0)
    walk_term = (dist_m / 100.0) * walk_w
    total = miss_term + walk_term

    if p_miss > 0.3:
        why = f"撲空風險高({p_miss:.0%})，後悔主要來自可能白跑"
    elif dist_m > 600:
        why = f"有位機率高({ra.p_available:.0%})但較遠，後悔主要來自步行"
    else:
        why = f"有位機率{ra.p_available:.0%}、距離適中，整體後悔低"
    if rain and dist_m > 400:
        why += "；雨天步行成本加權"

    return RegretResult(
        seg_id=ra.seg_id, name=name or ra.seg_id,
        expected_regret=total, miss_term=miss_term, walk_term=walk_term,
        p_available=ra.p_available, pred_avail=ra.avail_p50,
        avail_worst=ra.avail_p90, dist_m=dist_m, rationale=why,
    )


def value_of_information(ra: RiskAssessment, detour_m: int) -> dict:
    """
    資訊價值：若去現場確認那格(繞 detour_m)能消除不確定性，值不值得？
    粗略：不確定性越大(p_available 離 0/1 越遠) → 確認越有價值；繞路越遠 → 越不值。
    回傳 worth_checking 與一句白話建議。
    """
    uncertainty = 1.0 - abs(ra.p_available - 0.5) * 2  # 0(確定)~1(最不確定)
    voi = uncertainty * MISS_PENALTY                    # 確認最多能省下的期望撲空後悔
    cost = (detour_m / 100.0) * WALK_PENALTY_PER_100M
    worth = voi > cost
    if ra.p_available > 0.85:
        advice = f"有位機率{ra.p_available:.0%}已很高，不必繞路確認，直接賭"
    elif worth:
        advice = f"不確定性高，繞{detour_m}m去確認划算"
    else:
        advice = f"不確定但繞{detour_m}m去確認不划算，寧可賭把握較高的選項"
    return {"worth_checking": bool(worth), "voi": round(voi, 1),
            "check_cost": round(cost, 1), "advice": advice}


# singleton
_risk_table: Optional[RiskTable] = None


def get_risk_table() -> RiskTable:
    global _risk_table
    if _risk_table is None:
        _risk_table = RiskTable()
    return _risk_table
