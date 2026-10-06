"""SYSTEM_PROMPT 建構器 + Anthropic tools schema。"""
from __future__ import annotations
from typing import Any


TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "geocode_destination",
        "description": "把口語地名 (如『板橋車站』『新埔捷運站』) 轉成 lat/lon 座標。",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "get_weather_at_arrival",
        "description": "查指定座標、指定時間 (ISO 8601) 的天氣 (氣溫、濕度、降雨)。",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "at_iso": {"type": "string"},
            },
            "required": ["lat", "lon", "at_iso"],
        },
    },
    {
        "name": "list_candidate_segments",
        "description": "列出目的地座標附近指定半徑內的候選停車路段 (含座標、總車位數、收費等靜態資訊)。",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "radius_km": {"type": "number"},
                "top_k": {"type": "integer"},
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "predict_occupancy",
        "description": (
            "【核心工具】呼叫專案訓練好的 LightGBM 模型 (32 特徵, R²=0.96) 預測指定路段在指定時間的"
            "佔用率與剩餘車位。可一次傳多個 seg_id，並在同任務多次呼叫不同時間以掃描出發時機 "
            "(例如 now / now+15min / now+30min)。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "segment_ids": {"type": "array", "items": {"type": "string"}},
                "at_iso": {"type": "string"},
            },
            "required": ["segment_ids", "at_iso"],
        },
    },
    {
        "name": "get_live_tdx",
        "description": "查指定路段的即時剩餘車位 (TDX poller / 模擬模式從 recent_obs 取最近一筆)。",
        "input_schema": {
            "type": "object",
            "properties": {"seg_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["seg_ids"],
        },
    },
    {
        "name": "read_edge_sensor",
        "description": (
            "讀桌上實體 demo 板的真實邊緣感測器 (目前由 SensorSim 模擬: Pi 鏡頭 HSV + "
            "ESP32 超音波 兩源融合)。回傳 6 個車格的 free/occupied、信心度與最後變化時間。"
            "Watcher 偵測到變動會主動丟 [EDGE EVENT] 訊息給你。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "slot_id": {"type": "integer", "description": "若指定 1-6 只回該格；否則回全部"}
            },
        },
    },
    {
        "name": "get_route_eta",
        "description": "估算開車從起點到目的地的距離與時間。",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_lat": {"type": "number"},
                "from_lon": {"type": "number"},
                "to_lat": {"type": "number"},
                "to_lon": {"type": "number"},
            },
            "required": ["from_lat", "from_lon", "to_lat", "to_lon"],
        },
    },
    {
        "name": "decide_with_regret",
        "description": (
            "【決策核心】在已預測的候選路段中，用『期望後悔最小(argmin regret)』而非"
            "『預測最空(argmax)』挑選。撲空(到了沒位)的後悔遠大於多走幾步，所以一個"
            "『預測最空但 35% 機率撲空』的車位未必勝過『稍擠但 92% 有位』的。"
            "同時回傳資訊價值(VoI)：值不值得繞路去確認。"
            "candidates 請直接帶 predict_occupancy 回來的 predictions(各自補上 dist_m 距離)。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "seg_id": {"type": "string"},
                            "name": {"type": "string"},
                            "dist_m": {"type": "integer"},
                            "pred_occ": {"type": "number"},
                            "p_available": {"type": "number"},
                            "avail_likely": {"type": "integer"},
                            "avail_worst": {"type": "integer"},
                        },
                        "required": ["seg_id", "dist_m"],
                    },
                },
                "rain": {"type": "boolean", "description": "是否下雨(雨天步行成本加權)"},
                "at_iso": {"type": "string", "description": "抵達時間，請帶與 predict_occupancy 相同的時間"},
            },
            "required": ["candidates"],
        },
    },
    {
        "name": "lock_mission",
        "description": (
            "鎖定任務：選定主推路段 + 備案。會驅動 ESP32 OLED 顯示、LED 轉綠、TTS 播報、"
            "LINE 推播。呼叫後狀態自動轉 MONITORING。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "primary": {"type": "string"},
                "primary_name": {"type": "string"},
                "backups": {"type": "array", "items": {"type": "string"}},
                "arrival_iso": {"type": "string"},
                "strategy_note": {"type": "string"},
            },
            "required": ["primary", "primary_name", "backups", "arrival_iso", "strategy_note"],
        },
    },
    {
        "name": "alert_user",
        "description": "綜合致動：OLED 紅閃 + 蜂鳴器 + TTS 警示 + LINE 推播。REPLANNING 限定。",
        "input_schema": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["normal", "alert"]},
                "reason": {"type": "string"},
                "tts": {"type": "string", "description": "TTS 念出的繁體中文"},
                "line": {"type": "string", "description": "LINE 推播訊息"},
            },
            "required": ["severity", "reason", "tts", "line"],
        },
    },
]


TOOL_WHITELIST: dict[str, set[str]] = {
    "PLANNING":   {"geocode_destination", "get_weather_at_arrival",
                   "list_candidate_segments", "predict_occupancy",
                   "get_route_eta", "decide_with_regret", "lock_mission"},
    "MONITORING": {"read_edge_sensor", "get_live_tdx"},
    "REPLANNING": {"predict_occupancy", "get_live_tdx",
                   "decide_with_regret", "lock_mission", "alert_user"},
}


def tools_for(state: str) -> list[dict[str, Any]]:
    """Anthropic 格式 (歷史相容用)；目前主程式走 Gemini。"""
    allowed = TOOL_WHITELIST.get(state, set())
    return [t for t in TOOLS_SCHEMA if t["name"] in allowed]


# ── Gemini function-calling 格式轉換 ─────────────────────────────────────
_TYPE_MAP = {
    "string": "STRING", "integer": "INTEGER", "number": "NUMBER",
    "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT",
}


def _to_gemini_schema(s: Any) -> Any:
    """把 Claude/JSON-Schema 子節點轉成 Gemini Schema dict。"""
    if not isinstance(s, dict):
        return s
    out: dict[str, Any] = {}
    if "type" in s:
        t = s["type"]
        if isinstance(t, list):
            t = next((x for x in t if x != "null"), t[0])
        out["type"] = _TYPE_MAP.get(t, str(t).upper())
    if "description" in s:
        out["description"] = s["description"]
    if "properties" in s:
        out["properties"] = {k: _to_gemini_schema(v) for k, v in s["properties"].items()}
    if "required" in s and s["required"]:
        out["required"] = list(s["required"])
    if "items" in s:
        out["items"] = _to_gemini_schema(s["items"])
    if "enum" in s:
        out["enum"] = list(s["enum"])
    return out


def tools_for_gemini(state: str) -> list[dict[str, Any]] | None:
    """回傳 Gemini SDK 接受的 tools list (含 function_declarations)。"""
    allowed = TOOL_WHITELIST.get(state, set())
    decls: list[dict[str, Any]] = []
    for t in TOOLS_SCHEMA:
        if t["name"] not in allowed:
            continue
        params = _to_gemini_schema(t.get("input_schema", {"type": "object"}))
        # Gemini 要求 parameters 至少是 OBJECT
        if params.get("type") != "OBJECT":
            params = {"type": "OBJECT", "properties": {}}
        decls.append({
            "name": t["name"],
            "description": t["description"],
            "parameters": params,
        })
    if not decls:
        return None
    return [{"function_declarations": decls}]


def build_system_prompt(state: str, prefs: dict, sim_now_iso: str = "",
                         mission_summary: str = "") -> str:
    prefs_str = "尚無記錄" if not prefs else "; ".join(f"{k}={v}" for k, v in prefs.items())
    mission_block = f"【目前任務】 {mission_summary}\n" if mission_summary else ""

    return f"""你是 PARK-PILOT — 一個會自主規劃、主動呼叫工具、會看真實感測器、會自己 re-plan 的停車代理人。

【絕對規則】
1. 你目前處於三種狀態之一：PLANNING / MONITORING / REPLANNING；不同狀態能用的工具子集不同 (見下方)。
2. 每次輸出「必須」先用繁體中文白話講一段思考，把使用者目標、你打算做什麼、為什麼這樣做、哪些取捨講清楚。這段思考會被即時串流給評審看，所以要像在「自言自語」而不是寫報告。
3. 預測佔用率「禁止」憑空猜測，必須呼叫 predict_occupancy 工具取得 LightGBM 的真實預測。
4. 預測佔用率時呼叫 predict_occupancy（它使用 LightGBM 模型），思考中簡短說明你要預測哪些路段、哪個時間即可，不必誇飾。
5. PLANNING 時要呼叫 predict_occupancy 至少兩次以上不同時間 (例如抵達當下、抵達後 15 分、抵達後 30 分)，做出發時機掃描，補上單步預測弱點。
6. 【重要】拿到 predict_occupancy 的結果後，「不要」直接挑 pred_avail 最大的那個就鎖定。
   要呼叫 decide_with_regret，把候選(帶各自 dist_m、p_available、avail_likely)交給它，用「期望後悔最小」挑選。
   因為撲空(到了沒位)的代價遠大於多走幾步：一個「預測最空但可能撲空」的車位，未必勝過「稍擠但幾乎一定有位」的。
   在 thinking 要白話講出這個權衡，例如「A 預測空位最多但歷史上 30% 會客滿，B 雖然稍擠但 92% 有位、後悔更低，所以我選 B」。
   若 decide_with_regret 的 best 和 greedy_choice 不同，一定要說明「為什麼我不選看起來最空的那個」。
7. REPLANNING 時：先解釋為何要重排 → predict_occupancy(備案路段) → decide_with_regret → lock_mission 新主案 → 最後一定呼叫 alert_user(severity="alert") 通知使用者。
8. PLANNING 的最後一步是 lock_mission，呼叫完就停止輸出。
   REPLANNING 的最後一步是 alert_user(severity="alert")，要在 lock_mission 之後呼叫，然後停止輸出。
9. tool call 全輪上限 12 次；temperature=0.2，請務實精準。

【目前狀態】 {state}
【目前模擬時間 (sim_now)】 {sim_now_iso or "未設定"}
　— 使用者口語『七點』請理解成 sim_now 同日的 19:00；ISO 格式範例：{(sim_now_iso[:10] + 'T19:00:00') if sim_now_iso else '2026-04-21T19:00:00'}
{mission_block}
【可用工具子集】
PLANNING   → geocode_destination, get_weather_at_arrival, list_candidate_segments,
             predict_occupancy, get_route_eta, decide_with_regret, lock_mission
MONITORING → read_edge_sensor, get_live_tdx
REPLANNING → predict_occupancy, get_live_tdx, decide_with_regret, lock_mission, alert_user

【使用者偏好 (從過往任務累積)】
{prefs_str}

開始：直接用思考開頭，敘述你聽到的需求與接下來的計畫，再依序呼叫工具。"""
