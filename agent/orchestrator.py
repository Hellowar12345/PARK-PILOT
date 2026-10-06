"""
Gemini tool-use loop with async streaming.

差異備忘 (vs Anthropic Claude):
* `system_instruction` 不是 message，是 config 欄位
* messages 改稱 contents，role: "user" / "model" (不是 "assistant")
* tool call 是 model 訊息裡的 function_call part；
  tool result 是「user 訊息 + function_response part」
* finish_reason 永遠是 STOP，要看 parts 裡有沒有 function_call 判斷該不該繼續 loop
"""
from __future__ import annotations
import os
from typing import Any, AsyncIterator, Optional

from google import genai
from google.genai import types as gt

from .broker import broker
from .prompts import build_system_prompt, tools_for_gemini
from .state import session
from .tools import dispatch
from . import traces

MODEL = "gemini-3.1-flash-lite"
MAX_TOOL_CALLS = 12
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE = 0.2


_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY 未設定。請在 .env 設定 GEMINI_API_KEY=... 後重啟。"
            )
        _client = genai.Client(api_key=key)
    return _client


async def _emit(ev: dict[str, Any]) -> None:
    await broker.publish(ev)


async def run_session(initial_state: str, user_text: str,
                      prior_messages: Optional[list] = None) -> AsyncIterator[dict[str, Any]]:
    """
    跑一輪 PLANNING 或 REPLANNING session。
    initial_state: "PLANNING" | "REPLANNING"
    """
    session.state = initial_state
    await _emit({"type": "state_change", "state": initial_state})
    traces.log("session_start", {"state": initial_state, "user_text": user_text[:200]},
               mission_id=session.mission.mission_id if session.mission else "")

    mission_summary = ""
    if session.mission:
        m = session.mission
        mission_summary = (f"primary={m.primary_seg}({m.primary_name}), "
                           f"backups={m.backups}, arrival={m.arrival_iso}")

    contents: list[dict[str, Any]] = list(prior_messages or [])
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    tool_calls_used = 0
    client = _get_client()

    while True:
        if tool_calls_used >= MAX_TOOL_CALLS:
            ev = {"type": "agent_warning", "text": f"已達 tool call 上限 ({MAX_TOOL_CALLS})"}
            yield ev
            await _emit(ev)
            break

        sys_prompt = build_system_prompt(
            state=session.state,
            prefs=session.preferences,
            sim_now_iso=session.sim_now_iso,
            mission_summary=mission_summary,
        )
        tools = tools_for_gemini(session.state)

        function_calls: list[Any] = []
        finish_reason: Any = None
        # 保留 part 順序 + thought_signature (gemini-3.x function-calling 必要)
        assistant_parts: list[dict[str, Any]] = []

        try:
            cfg = gt.GenerateContentConfig(
                system_instruction=sys_prompt,
                tools=tools if tools else None,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            stream = await client.aio.models.generate_content_stream(
                model=MODEL,
                contents=contents,
                config=cfg,
            )
            async for chunk in stream:
                if not chunk.candidates:
                    continue
                cand = chunk.candidates[0]
                if cand.content and cand.content.parts:
                    for part in cand.content.parts:
                        entry: dict[str, Any] = {}
                        txt = getattr(part, "text", None)
                        fc = getattr(part, "function_call", None)
                        ts = getattr(part, "thought_signature", None)
                        if txt:
                            entry["text"] = txt
                            ev = {"type": "thinking", "text": txt}
                            yield ev
                            await _emit(ev)
                        if fc:
                            function_calls.append(fc)
                            entry["function_call"] = {
                                "name": fc.name,
                                "args": dict(fc.args or {}),
                            }
                            ev = {
                                "type": "tool_call_start",
                                "name": fc.name,
                                "id": getattr(fc, "id", None) or f"fc_{tool_calls_used + len(function_calls)}",
                            }
                            yield ev
                            await _emit(ev)
                        # thought_signature 一定要原封送回，否則下一輪 400
                        if ts is not None:
                            entry["thought_signature"] = ts
                        if "text" in entry or "function_call" in entry:
                            assistant_parts.append(entry)
                if cand.finish_reason:
                    finish_reason = cand.finish_reason
        except Exception as e:
            ev = {"type": "error", "text": f"Gemini API 錯誤: {e}"}
            yield ev
            await _emit(ev)
            return

        if assistant_parts:
            contents.append({"role": "model", "parts": assistant_parts})

        if not function_calls:
            ev = {"type": "agent_done", "stop_reason": str(finish_reason)}
            yield ev
            await _emit(ev)
            break

        # 執行工具，收集 function_response parts
        response_parts: list[dict[str, Any]] = []
        for fc in function_calls:
            tool_calls_used += 1
            fc_args = dict(fc.args or {})
            ev_call = {"type": "tool_call", "name": fc.name, "input": fc_args}
            yield ev_call
            await _emit(ev_call)
            traces.log("tool_call", {"name": fc.name, "input": fc_args},
                       mission_id=session.mission.mission_id if session.mission else "")

            result = await dispatch(fc.name, fc_args)

            ev_res = {"type": "tool_result", "name": fc.name, "result": result}
            yield ev_res
            await _emit(ev_res)
            traces.log("tool_result", {"name": fc.name, "result_preview": str(result)[:500]},
                       mission_id=session.mission.mission_id if session.mission else "")

            response_parts.append({
                "function_response": {"name": fc.name, "response": result}
            })

        contents.append({"role": "user", "parts": response_parts})

    # session 結束才轉 MONITORING — 讓 REPLAN 在 lock_mission 後仍能呼叫 alert_user
    if initial_state in ("PLANNING", "REPLANNING") and session.mission:
        session.state = "MONITORING"
        await _emit({"type": "state_change", "state": "MONITORING"})

    session.messages = contents
