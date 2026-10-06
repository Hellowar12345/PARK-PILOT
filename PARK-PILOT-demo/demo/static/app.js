"use strict";

// ── 地圖初始化（板橋車站） ─────────────────────────────────────────────
const map = L.map("map", { center: [25.014, 121.463], zoom: 14 });
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "© OpenStreetMap contributors",
  maxZoom: 19,
}).addTo(map);

// ── 全域狀態 ──────────────────────────────────────────────────────────
let destMarker = null;
let destCircle = null;
let resultMarkers = [];
let mode = "simulation";
let selectedCardIdx = -1;

// ── 模式切換 ────────────────────────────────────────────────────────────
function setMode(m) {
  mode = m;
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.mode === m));
  document.getElementById("sim-time-block").style.display = m === "simulation" ? "" : "none";
  if (m === "live") checkLiveStatus();
}

async function checkLiveStatus() {
  try {
    const r = await fetch("/api/live_status").then(r => r.json());
    const banner = document.getElementById("live-banner");
    if (!r.ready) {
      banner.style.display = "block";
      banner.textContent = `⚠️ 即時模式：正在蒐集資料 (${r.ready_segments}/${r.total_segments} 路段就緒)，請等 5~10 分鐘`;
    } else {
      banner.style.display = "none";
    }
  } catch (_) {}
}

// ── 點地圖 → 設目的地 ────────────────────────────────────────────────
map.on("click", (e) => {
  const { lat, lng } = e.latlng;
  placeDestMarker(lat, lng);
});

function placeDestMarker(lat, lng) {
  if (destMarker) map.removeLayer(destMarker);
  if (destCircle) map.removeLayer(destCircle);

  destMarker = L.marker([lat, lng], {
    icon: L.divIcon({ className: "dest-icon", iconSize: [18, 18], iconAnchor: [9, 9] }),
  }).addTo(map).bindPopup("📍 目的地").openPopup();

  const radiusM = parseInt(document.getElementById("radius").value);
  destCircle = L.circle([lat, lng], {
    radius: radiusM, color: "#dc2626", fillColor: "#dc2626", fillOpacity: 0.08, weight: 2,
  }).addTo(map);
}

// 半徑 slider 連動圓圈
document.getElementById("radius").addEventListener("input", function () {
  if (destCircle) destCircle.setRadius(parseInt(this.value));
});

// ── 執行預測 ────────────────────────────────────────────────────────────
async function doPredictFromMarker() {
  if (!destMarker) {
    alert("請先在地圖上點選目的地");
    return;
  }
  const { lat, lng } = destMarker.getLatLng();
  const radiusKm = parseInt(document.getElementById("radius").value) / 1000;
  const eta      = parseInt(document.getElementById("eta").value);
  const simNow   = mode === "simulation" ? document.getElementById("sim-now").value : null;

  const btn = document.getElementById("btn-predict");
  btn.disabled = true;
  btn.textContent = "預測中...";

  clearResults();

  try {
    const body = { lat, lon: lng, radius_km: radiusKm, eta_minutes: eta, mode };
    if (simNow) body.sim_now = simNow;

    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      alert("預測失敗：" + (err.detail || resp.status));
      return;
    }

    const results = await resp.json();
    renderResults(results, eta, simNow);
  } catch (e) {
    alert("連線錯誤：" + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "🔍 搜尋停車位";
  }
}

// ── 清除結果 ─────────────────────────────────────────────────────────────
function clearResults() {
  resultMarkers.forEach(m => map.removeLayer(m));
  resultMarkers = [];
  document.getElementById("result-list").innerHTML = "";
  document.getElementById("sidebar-hint").style.display = "none";
  selectedCardIdx = -1;
}

// ── 渲染結果 ─────────────────────────────────────────────────────────────
function renderResults(results, eta, simNow) {
  const list = document.getElementById("result-list");

  if (!results.length) {
    list.innerHTML = '<div class="sidebar-hint"><p>此範圍內無收費路段</p><p style="font-size:12px;color:#888">嘗試擴大搜尋半徑</p></div>';
    return;
  }

  const simLabel = simNow ? new Date(simNow).toLocaleString("zh-TW", { hour12: false }) : "現在";
  const arrLabel = simNow
    ? new Date(new Date(simNow).getTime() + eta * 60000).toLocaleString("zh-TW", { hour12: false })
    : `+${eta} 分後`;

  list.innerHTML = `<div class="result-header">找到 ${results.length} 個路段 &nbsp;·&nbsp; 抵達時間：${arrLabel}</div>`;

  results.forEach((r, idx) => {
    // ── Sidebar 卡片 ──
    const card = document.createElement("div");
    card.className = "card";
    card.id = `card-${idx}`;
    card.onclick = () => selectCard(idx, results);

    const predOcc  = r.predicted_occ;
    const curOcc   = r.current_occ;
    const actOcc   = r.actual_occ;
    const total    = r.TotalSpaces;

    const predColor = occupancyClass(predOcc);
    const curColor  = occupancyClass(curOcc);
    const actColor  = occupancyClass(actOcc);

    const fareStr = r.fare_price != null ? `💰 ${r.fare_price} 元/時` : "免費";
    const distStr = r.distance_m < 1000 ? `${r.distance_m} m` : `${(r.distance_m / 1000).toFixed(1)} km`;

    const actBlock = actOcc != null
      ? `<div class="occ-block ${actColor}">
           <div class="label">實際空位</div>
           <div class="value">${r.actual_avail ?? "?"}</div>
           <div class="sub">誤差 ${Math.abs((r.predicted_avail ?? 0) - (r.actual_avail ?? 0))} 位</div>
         </div>`
      : "";

    card.innerHTML = `
      <div class="card-rank">${idx + 1}</div>
      <div class="card-body">
        <div class="card-name">${r.name}</div>
        <div class="card-dist">🚶 ${distStr} &nbsp;·&nbsp; ${fareStr} &nbsp;·&nbsp; 共 ${total} 格</div>
        <div class="occ-bars">
          <div class="occ-block ${curColor}">
            <div class="label">現在空位</div>
            <div class="value">${r.current_avail ?? "?"}</div>
            <div class="sub">${pct(curOcc)}</div>
          </div>
          <div class="occ-block ${predColor}">
            <div class="label">抵達預測</div>
            <div class="value">${r.predicted_avail}</div>
            <div class="sub">${pct(predOcc)}</div>
          </div>
          ${actBlock}
        </div>
        <span class="fare-tag">${fareStr}</span>
      </div>`;

    list.appendChild(card);

    // ── 地圖 marker ──
    const markerColor = ["#19a34a", "#d97706", "#dc2626"][
      predOcc < 0.7 ? 0 : predOcc < 0.9 ? 1 : 2
    ];
    const icon = L.divIcon({
      className: "",
      html: `<div style="background:${markerColor};color:#fff;border-radius:50%;width:30px;height:30px;
        display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;
        border:2px solid #fff;box-shadow:0 2px 5px rgba(0,0,0,.4)">${r.predicted_avail}</div>`,
      iconSize: [30, 30], iconAnchor: [15, 15],
    });

    const marker = L.marker([r.lat, r.lon], { icon })
      .addTo(map)
      .bindPopup(popupHtml(r, idx + 1, predColor));
    marker.on("click", () => selectCard(idx, results));
    resultMarkers.push(marker);
  });
}

// ── 選中卡片 ─────────────────────────────────────────────────────────────
function selectCard(idx, results) {
  if (selectedCardIdx >= 0) {
    document.getElementById(`card-${selectedCardIdx}`)?.classList.remove("selected");
  }
  selectedCardIdx = idx;
  document.getElementById(`card-${idx}`)?.classList.add("selected");
  document.getElementById(`card-${idx}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });

  const r = results[idx];
  if (resultMarkers[idx]) {
    map.setView([r.lat, r.lon], 16);
    resultMarkers[idx].openPopup();
  }
}

// ── 輔助函式 ─────────────────────────────────────────────────────────────
function occupancyClass(occ) {
  if (occ == null) return "";
  return occ < 0.7 ? "green" : occ < 0.9 ? "yellow" : "red";
}

function pct(occ) {
  return occ != null ? `佔用 ${Math.round(occ * 100)}%` : "-";
}

function popupHtml(r, rank, colorClass) {
  return `
    <b>#${rank} ${r.name}</b><br>
    <span class="popup-label">抵達時預測空位</span><br>
    <span class="popup-avail ${colorClass}" style="color:${colorClass === 'green' ? '#19a34a' : colorClass === 'yellow' ? '#d97706' : '#dc2626'}">${r.predicted_avail} 格</span>
    <span style="color:#888;font-size:11px">/ ${r.TotalSpaces} 格</span><br>
    <span class="popup-label">現在：${r.current_avail ?? "?"} 格 &nbsp;·&nbsp; ${r.fare_price != null ? r.fare_price + " 元/時" : "免費"}</span>
    ${r.actual_occ != null ? `<br><span class="popup-label">實際：${r.actual_avail ?? "?"} 格（誤差 ${Math.abs((r.predicted_avail ?? 0) - (r.actual_avail ?? 0))} 格）</span>` : ""}
  `;
}

// ── 初始化：嘗試設定預設 sim_now 為最新觀測 ────────────────────────────
(async () => {
  try {
    const ts = await fetch("/api/sim_timestamps").then(r => r.json());
    if (ts.length) {
      const latest = ts[0].replace("T", "T"); // ISO datetime-local
      document.getElementById("sim-now").value = latest.slice(0, 16);
    }
  } catch (_) {}
})();
