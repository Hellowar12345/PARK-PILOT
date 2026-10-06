"use strict";

// ── 全域 DOM 參照 ─────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const pill = $("state-pill");
const thinking = $("thinking-area");
const toolcalls = $("toolcalls");
const slotsGrid = $("slots-grid");
const alertLog = $("alert-log");
const missionBody = $("mission-body");
const ledRing = $("led-ring");

// ── 共用工具 ─────────────────────────────────────────────────────────
function setState(s) {
  pill.dataset.state = s;
  pill.textContent = s;
}

function appendThinking(text) {
  // 第一次有 thinking 內容時清掉 placeholder
  const ph = thinking.querySelector(".thinking-placeholder");
  if (ph) ph.remove();
  thinking.append(document.createTextNode(text));
  thinking.scrollTop = thinking.scrollHeight;
}

function clearThinking() {
  thinking.innerHTML = '<div class="thinking-placeholder">思考中…</div>';
}

function addToolCard(name, input) {
  const card = document.createElement("div");
  card.className = `toolcall-card ${name}`;
  card.dataset.name = name;
  card.innerHTML = `
    <div class="tc-name">${name}</div>
    <div class="tc-args">${formatArgs(input)}</div>
    <div class="tc-result" style="display:none"></div>
  `;
  toolcalls.appendChild(card);
  toolcalls.scrollTop = toolcalls.scrollHeight;
  return card;
}

function formatArgs(input) {
  if (!input) return "";
  try {
    const s = JSON.stringify(input);
    return s.length > 200 ? s.slice(0, 200) + "…" : s;
  } catch { return String(input); }
}

function setToolResult(name, result) {
  // find last card with this name that hasn't been filled
  const cards = toolcalls.querySelectorAll(`.toolcall-card[data-name="${name}"]`);
  for (let i = cards.length - 1; i >= 0; i--) {
    const r = cards[i].querySelector(".tc-result");
    if (r && r.style.display === "none") {
      r.style.display = "block";
      r.innerHTML = "→ " + summarizeResult(name, result);
      return;
    }
  }
}

function summarizeResult(name, r) {
  if (!r) return "(no result)";
  if (r.ok === false) return `<span style="color:#ff4b5c">✗ ${r.error || "錯誤"}</span>`;
  switch (name) {
    case "geocode_destination":  return `${r.name} (${r.lat?.toFixed(4)}, ${r.lon?.toFixed(4)})`;
    case "get_weather_at_arrival": return `${r.temperature}°C / ${r.humidity}% / ${r.condition || ""}`;
    case "list_candidate_segments": return `${r.count} 個路段 (top: ${r.segments?.slice(0,3).map(s=>s.seg_id).join(", ")}…)`;
    case "predict_occupancy": {
      const ps = (r.predictions || []).slice(0, 4)
        .map(p => `${p.seg_id}: ${(p.pred_occ * 100).toFixed(0)}% (空 ${p.pred_avail})`).join(", ");
      return `${r.model || "LightGBM"} @ ${r.at_iso} — ${ps}${r.count > 4 ? "…" : ""}`;
    }
    case "get_live_tdx":         return `${r.count} 路段 TDX 即時`;
    case "read_edge_sensor": {
      if (r.slots) {
        const occ = r.slots.filter(s => s.state === "occupied").map(s => `slot_${s.slot_id}`);
        return `target=slot_${r.target_slot} · 已佔: ${occ.join(",") || "(無)"}`;
      }
      return r.slot ? `slot_${r.slot.slot_id}: ${r.slot.state} (conf ${r.slot.confidence})` : "(空)";
    }
    case "get_route_eta":        return `${r.distance_km} km · 約 ${r.eta_min} 分`;
    case "lock_mission":         return `✓ 已鎖定 ${r.mission_id}`;
    case "alert_user":           return `✓ 已通知`;
    default: return JSON.stringify(r).slice(0, 120);
  }
}

// ── SSE event handler ────────────────────────────────────────────────
function attachEvents() {
  const es = new EventSource("/agent/events");
  es.onmessage = (e) => {
    try { handleEvent(JSON.parse(e.data)); }
    catch (err) { console.warn("bad event", e.data, err); }
  };
  es.onopen = () => {
    if (sseRetry > 0) appendThinking("\n[連線已恢復]\n");
    sseRetry = 0;
  };
  es.onerror = () => {
    es.close();
    if (sseRetry >= 5) {
      appendThinking("\n[連線中斷，請手動重新整理頁面]\n");
      return;
    }
    const wait = Math.min(8000, 2000 * Math.pow(2, sseRetry));
    sseRetry++;
    appendThinking(`\n[連線中斷，${Math.round(wait / 1000)} 秒後重新連線 (${sseRetry}/5)…]\n`);
    setTimeout(attachEvents, wait);
  };
}
let sseRetry = 0;

function handleEvent(ev) {
  switch (ev.type) {
    case "init":
      if (ev.state) setState(ev.state);
      if (ev.sensors) renderSlots(ev.sensors);
      if (ev.display) renderDisplay(ev.display);
      if (ev.mission) renderMission(ev.mission);
      if (ev.sim_now_iso) $("sim-now").value = ev.sim_now_iso.slice(0, 16);
      break;
    case "state_change":
      setState(ev.state);
      if (ev.state === "PLANNING" || ev.state === "REPLANNING") {
        clearThinking();
        if (ev.state === "REPLANNING") {
          toolcalls.innerHTML = ""; // fresh round
        }
      }
      break;
    case "thinking":
      appendThinking(ev.text);
      break;
    case "tool_call_start":
      addToolCard(ev.name, {});
      break;
    case "tool_call":
      // backfill the args into the most recent matching card
      const cards = toolcalls.querySelectorAll(`.toolcall-card[data-name="${ev.name}"]`);
      const last = cards[cards.length - 1];
      if (last) last.querySelector(".tc-args").textContent = formatArgs(ev.input);
      else addToolCard(ev.name, ev.input);
      break;
    case "tool_result":
      setToolResult(ev.name, ev.result);
      break;
    case "sensor_event":
      // optimistic UI already updated; this confirms
      flashSlot(ev.slot_id, ev.next);
      break;
    case "display_update":
      renderDisplay(ev);
      break;
    case "alert":
      renderDisplay(ev);
      addAlertItem(ev);
      // 嘗試瀏覽器內 TTS 模擬 ESP32 TTS
      speak(ev.tts || ev.reason);
      break;
    case "mission_locked":
      renderMission(ev.mission);
      break;
    case "edge_event_detected":
      addAlertItem({ reason: `偵測到目標 slot_${ev.slot} 被佔 (conf ${ev.confidence})`, tts: "", line: "" });
      break;
    case "replan_skipped":
      addAlertItem({ reason: `Re-plan 跳過: ${ev.reason}`, tts: "", line: "" });
      break;
    case "reset":
      handleReset();
      break;
    case "agent_done":
      // ok
      break;
    case "agent_warning":
      appendThinking(`\n[警告] ${ev.text}\n`);
      break;
    case "error":
      appendThinking(`\n\n[錯誤] ${ev.text}\n`);
      break;
    default:
      // console.debug("event", ev);
  }
}

// ── 渲染 ──────────────────────────────────────────────────────────────
function renderSlots(sensors) {
  const slots = sensors.slots || [];
  const target = sensors.target_slot;
  slotsGrid.innerHTML = "";
  slots.forEach(s => {
    const div = document.createElement("div");
    div.className = `slot ${s.state} ${s.slot_id === target ? "target" : ""}`;
    div.dataset.slotId = s.slot_id;
    div.innerHTML = `
      <div class="num">${s.slot_id}</div>
      <div class="label">${s.state}</div>
    `;
    div.onclick = (e) => toggleSlot(s.slot_id);
    div.oncontextmenu = (e) => { e.preventDefault(); setTarget(s.slot_id); };
    slotsGrid.appendChild(div);
  });
}

function flashSlot(slot_id, state) {
  const el = slotsGrid.querySelector(`.slot[data-slot-id="${slot_id}"]`);
  if (!el) return;
  el.classList.remove("free", "occupied");
  el.classList.add(state);
  el.querySelector(".label").textContent = state;
  el.style.transition = "transform .3s";
  el.style.transform = "scale(1.1)";
  setTimeout(() => { el.style.transform = "scale(1)"; }, 200);
}

function renderDisplay(d) {
  if (d.oled_line1 !== undefined) $("oled-line1").textContent = d.oled_line1;
  if (d.oled_line2 !== undefined) $("oled-line2").textContent = d.oled_line2;
  if (d.oled_line3 !== undefined) $("oled-line3").textContent = d.oled_line3;
  if (d.led_color !== undefined) {
    ledRing.dataset.led = d.led_color;
    $("led-state").textContent = "LED: " + d.led_color;
  }
}

function renderMission(m) {
  if (!m) {
    missionBody.className = "mission-empty";
    missionBody.textContent = "尚未鎖定任務";
    return;
  }
  missionBody.className = "mission-locked";
  const arrival = (m.arrival_iso || "").replace("T", " ").slice(0, 16);
  missionBody.innerHTML = `
    <div class="ml-primary">${m.primary_name} <span style="font-weight:400;color:var(--text-dim)">(${m.primary_seg})</span></div>
    <div class="ml-meta">抵達 ${arrival} · ID ${m.mission_id}</div>
    <div class="ml-backups">備案: ${(m.backups || []).join(", ") || "—"}</div>
    <div class="ml-note">${m.strategy_note || ""}</div>
  `;
}

function addAlertItem(a) {
  if (alertLog.querySelector(".alert-empty")) alertLog.innerHTML = "";
  const div = document.createElement("div");
  div.className = "alert-item";
  div.innerHTML = `
    <div class="ai-reason">⚠ ${a.reason || a.severity || "alert"}</div>
    ${a.tts ? `<div class="ai-tts">🔊 "${a.tts}"</div>` : ""}
    ${a.line ? `<div class="ai-line">📲 LINE: ${a.line}</div>` : ""}
  `;
  alertLog.prepend(div);
  while (alertLog.children.length > 8) alertLog.lastChild.remove();
}

// ── 動作 ──────────────────────────────────────────────────────────────
async function toggleSlot(slot_id) {
  await fetch("/agent/sim/slot/toggle", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ slot_id }),
  });
}

async function setTarget(slot_id) {
  await fetch("/agent/sim/target", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ slot_id }),
  });
  // 重新讀 sensors 以更新 ⭐
  const st = await fetch("/agent/state").then(r => r.json());
  renderSlots(st.sensors);
}

async function dropLegoOnTarget() {
  // get current target
  const st = await fetch("/agent/state").then(r => r.json());
  const tgt = st.sensors.target_slot;
  await fetch("/agent/sim/slot/occupy", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ slot_id: tgt }),
  });
}

async function sendChat() {
  const inp = $("chat-input");
  const text = inp.value.trim();
  if (!text) return;
  inp.value = "";
  await streamChat(text);
}

function sendPreset(text) {
  $("chat-input").value = text;
  streamChat(text);
}

async function streamChat(text) {
  toolcalls.innerHTML = "";
  clearThinking();
  // 不依賴 /agent/events 也能即時看到（雙路徑：events SSE 也會收到一份）
  const resp = await fetch("/agent/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!resp.ok) {
    appendThinking(`\n[HTTP ${resp.status}] ${await resp.text()}\n`);
    return;
  }
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
      if (!chunk.startsWith("data:")) continue;
      const data = chunk.slice(5).trim();
      if (!data) continue;
      try { handleEvent(JSON.parse(data)); } catch (_) {}
    }
  }
}

async function resetAll() {
  await fetch("/agent/reset", { method: "POST" });
}

async function setSimTime() {
  const v = $("sim-now").value;
  if (!v) return;
  await fetch("/agent/sim/time", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ iso: v + ":00" }),
  });
}

function handleReset() {
  setState("IDLE");
  toolcalls.innerHTML = "";
  clearThinking();
  renderMission(null);
  alertLog.innerHTML = "";
}

function speak(text) {
  if (!text) return;
  try {
    if (!("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "zh-TW";
    u.rate = 1.05;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  } catch (_) {}
}

// ── 初始化 ────────────────────────────────────────────────────────────
(async () => {
  // 抓初始狀態 (含 sim_now、slots、mission)
  try {
    const st = await fetch("/agent/state").then(r => r.json());
    if (st.sim_now_iso) $("sim-now").value = st.sim_now_iso.slice(0, 16);
    setState(st.state);
    renderSlots(st.sensors);
    renderDisplay(st.display);
    renderMission(st.mission);
  } catch (e) {
    console.warn("init state failed", e);
  }
  attachEvents();
})();
