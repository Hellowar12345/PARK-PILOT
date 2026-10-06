"""
多智能體停車賽局模擬 (PARK-PILOT 想法升級的核心)。

中心命題（自我推翻預言 / minority game）：
    當每個駕駛都看到「同一個準確預測」說某條街最空，大家會一起湧過去 →
    那條街瞬間被塞爆 → 準確的預測親手摧毀了自己的準確度。
    這種稀缺與崩潰不存在於歷史資料（過去不是人人都用同一個 app），只能從競爭中湧現。

兩個關鍵洞見（模擬會親自驗證，不是嘴上說）：
    1. 全員貪心(naive)照同一個預測走 → 羊群效應 → 集體崩潰（很多人撲空）。
    2. 同質的理性 agent 無法自己協調（全員反身也只是把羊群換條街）。真正能改善的是：
       (a) 中央協調分流（coordinated）：PARK-PILOT 當車隊調度，把車分散到各街 → 近乎零撲空。
       (b) 少數偏離者得利（minority reflexive）：在貪心人群中，少數「預期羊群、反其道而行」的
           駕駛，個人撲空率明顯低於從眾的多數 —— 這就是 El Farol bar / minority game。

模擬為確定性（給定 seed），方便影片重現。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GameSegment:
    seg_id: str
    name: str
    capacity: int          # 本場開放給車隊競爭的車位數
    dist_m: int            # 到目的地的步行距離
    pred_free: int         # 公開的「預測剩餘車位」訊號（大家都看得到、且準確）
    occupied: int = 0

    @property
    def free_now(self) -> int:
        return max(0, self.capacity - self.occupied)


@dataclass
class DriverOutcome:
    driver_id: int
    strategy: str
    parked_seg: Optional[str] = None
    parked_name: str = ""
    walk_m: int = 0
    reroutes: int = 0
    tried: list[str] = field(default_factory=list)
    succeeded: bool = False


def _summarize(label: str, segs: list[GameSegment],
               outcomes: list[DriverOutcome]) -> dict:
    n = len(outcomes)
    parked = sum(1 for o in outcomes if o.succeeded)
    walks = [o.walk_m for o in outcomes if o.succeeded]
    return {
        "strategy_label": label,
        "n_drivers": n,
        "parked": parked,
        "missed": n - parked,
        "total_reroutes": sum(o.reroutes for o in outcomes),
        "avg_walk_m": round(sum(walks) / len(walks), 1) if walks else 0.0,
        "park_rate": round(parked / n, 3) if n else 0.0,
        "seg_final": [{"seg_id": s.seg_id, "name": s.name, "capacity": s.capacity,
                       "occupied": s.occupied, "pred_free": s.pred_free,
                       "dist_m": s.dist_m} for s in segs],
        "outcomes": [{"driver_id": o.driver_id, "strategy": o.strategy,
                      "parked_seg": o.parked_seg, "walk_m": o.walk_m,
                      "reroutes": o.reroutes, "succeeded": o.succeeded,
                      "tried": list(o.tried)} for o in outcomes],
    }


def _fresh(segments: list[GameSegment]) -> list[GameSegment]:
    return [GameSegment(s.seg_id, s.name, s.capacity, s.dist_m, s.pred_free, 0)
            for s in segments]


# ── 個別駕駛的偏好排序 ────────────────────────────────────────────────────

def _rank_naive(segs: list[GameSegment]) -> list[str]:
    """貪心：純看公開預測剩餘車位多者優先（人人算出同一答案 → 羊群）。"""
    return [s.seg_id for s in sorted(segs, key=lambda s: (-s.pred_free, s.dist_m))]


def _rank_reflexive(segs: list[GameSegment], n_competitors: int) -> list[str]:
    """
    反身推理：預期『越被預測為空的街，越多從眾者會去』。
    expected_crowd(seg) ∝ pred_free 佔比 × 競爭者數；
    挑「預期淨可得 = pred_free − expected_crowd」最高者，避開會被擠爆的熱門街。
    步行距離只當極小的次要 tiebreak（不可蓋過競爭訊號）。
    """
    attract = {s.seg_id: max(0.0, float(s.pred_free)) for s in segs}
    total = sum(attract.values()) or 1.0
    scored = []
    for s in segs:
        exp_crowd = n_competitors * (attract[s.seg_id] / total)
        net = s.pred_free - exp_crowd
        score = net - 0.0005 * s.dist_m   # 距離僅微弱 tiebreak
        scored.append((s.seg_id, score))
    scored.sort(key=lambda x: -x[1])
    return [sid for sid, _ in scored]


def _resolve(segs: list[GameSegment], order_per_driver: list[list[str]],
             strat_per_driver: list[str], max_tries: Optional[int] = None) -> list[DriverOutcome]:
    """
    依抵達順序，每位駕駛照自己的偏好排序嘗試停車，滿了就改道試下一個。
    max_tries：耐心上限 —— 繞 N 條街都滿就放棄離開（模擬真實「找不到車位」）。None=無限。
    """
    by_id = {s.seg_id: s for s in segs}
    outcomes = []
    for d, (order, strat) in enumerate(zip(order_per_driver, strat_per_driver)):
        oc = DriverOutcome(driver_id=d, strategy=strat)
        attempts = order if max_tries is None else order[:max_tries]
        for sid in attempts:
            oc.tried.append(sid)
            seg = by_id[sid]
            if seg.free_now > 0:
                seg.occupied += 1
                oc.parked_seg = sid
                oc.parked_name = seg.name
                oc.walk_m = seg.dist_m
                oc.succeeded = True
                oc.reroutes = len(oc.tried) - 1
                break
        else:
            oc.reroutes = len(oc.tried)   # 試完都滿 → 放棄（succeeded=False）
        outcomes.append(oc)
    return outcomes


# ── 三種世界 ──────────────────────────────────────────────────────────────

def run_selfish(segments: list[GameSegment], n_drivers: int,
                max_tries: Optional[int] = None) -> dict:
    """世界 A：全員貪心。所有人照同一個預測衝最空那條 → 羊群崩潰。
    max_tries：耐心上限（繞幾條街找不到就放棄）。"""
    segs = _fresh(segments)
    order = _rank_naive(segs)
    outcomes = _resolve(segs, [order] * n_drivers, ["naive"] * n_drivers,
                        max_tries=max_tries)
    return _summarize("世界A · 各自為政（全員照預測衝最空）", segs, outcomes)


def run_coordinated_blind(segments: list[GameSegment], n_drivers: int) -> dict:
    """
    世界 B'（資訊對等版）：協調者只用『公開預測 pred_free』做一次性 round-robin 分配，
    完全不讀任何即時佔用狀態 —— 與各自為政方掌握的資訊「完全相同」。
    用途：證明協調的優勢來自『集中分配（避免大家選同一個）』，而非『情報優勢』。
    """
    segs = _fresh(segments)
    outcomes = []
    plan = {s.seg_id: 0 for s in segs}              # 調度者的計畫（只用 pred_free 當每街上限）
    ring = sorted(segs, key=lambda s: (-s.pred_free, s.dist_m))
    for d in range(n_drivers):
        target = next((s for s in ring if plan[s.seg_id] < s.pred_free), None)
        oc = DriverOutcome(driver_id=d, strategy="coordinated_blind",
                           tried=[target.seg_id] if target else [])
        if target is not None:
            plan[target.seg_id] += 1
            target.occupied += 1
            oc.parked_seg = target.seg_id
            oc.parked_name = target.name
            oc.walk_m = target.dist_m
            oc.succeeded = True
        outcomes.append(oc)
    return _summarize("世界B' · 資訊對等協調（只用預測，不讀即時）", segs, outcomes)


def run_coordinated(segments: list[GameSegment], n_drivers: int) -> dict:
    """
    世界 B：中央協調（PARK-PILOT 當車隊調度）。
    貪心 water-filling：每台車派給『目前剩餘最多』的街 → 自然分散、先填滿才溢出。
    在總車位足夠時可達零撲空 —— 證明協調打敗各自為政。
    """
    segs = _fresh(segments)
    by_id = {s.seg_id: s for s in segs}
    outcomes = []
    for d in range(n_drivers):
        # 派給目前 free 最多的街（距離當 tiebreak）
        target = max(segs, key=lambda s: (s.free_now, -s.dist_m))
        oc = DriverOutcome(driver_id=d, strategy="coordinated", tried=[target.seg_id])
        if target.free_now > 0:
            target.occupied += 1
            oc.parked_seg = target.seg_id
            oc.parked_name = target.name
            oc.walk_m = target.dist_m
            oc.succeeded = True
        outcomes.append(oc)
    return _summarize("世界B · PARK-PILOT 協調分流", segs, outcomes)


def run_minority(segments: list[GameSegment], n_drivers: int,
                 reflexive_frac: float = 0.25,
                 max_tries: Optional[int] = None) -> dict:
    """
    世界 C：貪心人群中，少數『反身推理』者。
    驗證 minority game：少數預期羊群、反其道而行的駕駛，個人撲空率低於從眾多數。
    確定性散佈反身者於抵達序列中。
    """
    segs = _fresh(segments)
    k = max(1, int(round(n_drivers * reflexive_frac)))
    step = n_drivers / k
    reflexive_ids = {int(i * step) for i in range(k)}

    naive_order = _rank_naive(segs)
    # 反身者假設『其他人』(n-1) 都會從眾。
    # 邊界：此假設只在反身者為少數時成立；反身者佔比一高，大家都反著想 → 假設破裂、
    # 反身優勢收斂甚至反轉（見 build_sensitivity_charts.py 的 reflexive_frac 掃描）。
    reflexive_order = _rank_reflexive(segs, n_drivers - 1)

    orders, strats = [], []
    for d in range(n_drivers):
        if d in reflexive_ids:
            orders.append(reflexive_order); strats.append("reflexive")
        else:
            orders.append(naive_order); strats.append("naive")
    outcomes = _resolve(segs, orders, strats, max_tries=max_tries)

    # 分群統計：反身 vs 從眾 的個人表現
    def grp(name):
        g = [o for o in outcomes if o.strategy == name]
        if not g:
            return {"n": 0}
        return {
            "n": len(g),
            "park_rate": round(sum(o.succeeded for o in g) / len(g), 3),
            "avg_reroutes": round(sum(o.reroutes for o in g) / len(g), 2),
            "avg_walk_m": round(sum(o.walk_m for o in g if o.succeeded) /
                                max(1, sum(o.succeeded for o in g)), 1),
        }
    res = _summarize(f"世界C · {int(reflexive_frac*100)}% 反身推理混入貪心人群", segs, outcomes)
    res["by_group"] = {"reflexive": grp("reflexive"), "naive": grp("naive")}
    return res
