"use strict";
const $ = id => document.getElementById(id);

const LANDMARKS = {
  "板橋車站": [25.014, 121.4631], "新埔捷運站": [25.0359, 121.4685],
  "府中捷運站": [25.0095, 121.4595], "江子翠捷運站": [25.0303, 121.4761],
  "三重": [25.0613, 121.4858], "中和": [24.9988, 121.4990],
};

let GAME = null;       // 後端回傳的場景 + 三世界結果
let playing = false;

// ── 載入場景 ──────────────────────────────────────────────────────────
async function runScenario() {
  const dest = $("dest").value.trim();
  const [lat, lon] = LANDMARKS[dest] || LANDMARKS["板橋車站"];
  const nd = parseInt($("ndrivers").value) || null;
  const patience = parseInt($("patience").value) || 2;

  $("scenario-bar").innerHTML = "⏳ 呼叫 LightGBM 預測各路段、建立競爭場景中…";
  const resp = await fetch("/agent/game/run", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lat, lon, n_drivers: nd, patience }),
  });
  const r = await resp.json();
  if (!r.ok) { $("scenario-bar").innerHTML = "⚠ " + (r.error || "場景建立失敗"); return; }
  GAME = r;
  renderScenario();
}

function renderScenario() {
  const sc = GAME.scenario;
  const segList = sc.segments.map(s => `${s.name.slice(0,8)}(${s.capacity})`).join("、");
  $("scenario-bar").innerHTML =
    `真實新北路段 × LightGBM 預測　|　<b>${sc.n_drivers}</b> 台車競爭 <b>${sc.total_capacity}</b> 個車位　|　` +
    `路段(預測可停)：${segList}`;
  // 初始化兩邊停車場(空)
  buildLots("lots-a", sc.segments, false);
  buildLots("lots-b", sc.segments, true);
  resetScores();
  $("road-a").innerHTML = ""; $("road-b").innerHTML = "";
  $("caption").className = "g-caption";
  $("caption").innerHTML = "同一批車、同一個準確的預測。左邊各自為政，右邊交給 PARK-PILOT 協調。按「播放對照」。";
  $("btn-play").disabled = false;
}

function buildLots(containerId, segs, pilot) {
  const c = $(containerId); c.innerHTML = "";
  // 標記預測最空(最熱門)那條
  const maxFree = Math.max(...segs.map(s => s.pred_free));
  segs.forEach(s => {
    const popular = s.pred_free === maxFree;
    const lot = document.createElement("div");
    lot.className = "lot" + (popular ? " popular" : "");
    lot.id = `${containerId}-${s.seg_id}`;
    lot.innerHTML = `
      <div class="lot-head">
        <span class="lot-name">${s.name.slice(0,12)}${popular ? '<span class="tag">★預測最空</span>' : ''}</span>
        <span class="lot-meta">${s.dist_m}m</span>
      </div>
      <div class="lot-cells">${Array.from({length: s.capacity}, () => '<div class="cell"></div>').join("")}</div>
      <div class="lot-pred">預測可停 ${s.pred_free}</div>`;
    c.appendChild(lot);
  });
}

function resetScores() {
  ["a","b"].forEach(w => { $(`${w}-parked`).textContent = "0";
    $(`${w}-gaveup`).textContent = "0"; $(`${w}-reroute`).textContent = "0"; });
}

// ── 播放對照(兩世界同步動畫) ─────────────────────────────────────────
async function play() {
  if (!GAME || playing) return;
  playing = true; $("btn-play").disabled = true;
  renderScenario();  // reset visuals
  await sleep(300);

  const A = GAME.world_a_selfish, B = GAME.world_b_coordinated;
  const segA = GAME.scenario.segments, segB = GAME.scenario.segments;

  // 重置每格佔用計數
  const fillA = {}, fillB = {};
  segA.forEach(s => { fillA[s.seg_id] = 0; fillB[s.seg_id] = 0; });
  let aP=0,aG=0,aR=0, bP=0,bG=0,bR=0;

  $("caption").className = "g-caption";
  $("caption").innerHTML = "車隊抵達中… 兩邊看到的是<b>同一個</b> LightGBM 預測。";

  const N = GAME.scenario.n_drivers;
  const step = N > 30 ? 90 : 150;  // 車多就加快

  // 找「預測最空」那條街的名字(羊群目標)
  const segs = GAME.scenario.segments;
  const popular = segs.reduce((a, s) => s.pred_free > a.pred_free ? s : a, segs[0]);
  const midBeat = Math.floor(N * 0.45);

  for (let i = 0; i < N; i++) {
    // 中段旁白:點出羊群正在形成
    if (i === midBeat) {
      $("caption").className = "g-caption";
      $("caption").innerHTML =
        `左邊：大家都算出「<b>${popular.name.slice(0,10)}</b>」預測最空 → 一起衝過去。` +
        `右邊：PARK-PILOT 預期到這點，把車隊<b>分散</b>開。`;
    }
    const oa = A.outcomes[i], ob = B.outcomes[i];
    // 左：各自為政
    animateCar("road-a", oa, false);
    if (oa.succeeded) { fillA[oa.parked_seg]++; fillCell("lots-a", oa.parked_seg, fillA[oa.parked_seg], false); aP++; }
    else aG++;
    aR += oa.reroutes;
    // 右：協調
    animateCar("road-b", ob, true);
    if (ob.succeeded) { fillB[ob.parked_seg]++; fillCell("lots-b", ob.parked_seg, fillB[ob.parked_seg], true); bP++; }
    else bG++;
    bR += ob.reroutes;

    $("a-parked").textContent = aP; $("a-gaveup").textContent = aG; $("a-reroute").textContent = aR;
    $("b-parked").textContent = bP; $("b-gaveup").textContent = bG; $("b-reroute").textContent = bR;
    await sleep(step);
  }

  // 結局：標記「空著卻被浪費」的街 + 殺手字幕
  const wasted = GAME.headline.wasted_empty_streets || [];
  wasted.forEach(w => {
    const seg = segA.find(s => s.name === w.name);
    if (seg) $(`lots-a-${seg.seg_id}`)?.classList.add("emptied-wasted");
  });

  await sleep(400);
  const h = GAME.headline;
  const wastedSpots = wasted.reduce((a,w)=>a+w.free,0);
  $("caption").className = "g-caption punch";
  if (wasted.length && h.selfish_gave_up > 0) {
    $("caption").innerHTML =
      `各自為政：<b>${h.selfish_gave_up}</b> 人放棄離開、繞圈 <b>${h.selfish_reroutes}</b> 次` +
      `　——　卻有 <b>${wasted.length}</b> 條街、共 <b>${wastedSpots}</b> 個車位整條空著沒人去。` +
      `　準確的預測，因為人人都信，親手害大家撲空。`;
  } else {
    $("caption").innerHTML =
      `各自為政：繞圈 <b>${h.selfish_reroutes}</b> 次、<b>${h.selfish_gave_up}</b> 人放棄　vs　PARK-PILOT 協調：繞圈 <b>${h.coordinated_reroutes}</b> 次。`;
  }
  await sleep(2600);
  $("caption").className = "g-caption win";
  $("caption").innerHTML =
    `同一批車、同樣的車位：PARK-PILOT 協調分流 → 放棄離開 <b>${h.coordinated_gave_up}</b> 人、繞圈 <b>${h.coordinated_reroutes}</b> 次。` +
    `　差別不在預測多準，而在<b>有沒有人協調</b>。`;

  // ── 第三幕：少數反身推理者(minority game) — 條件式、誠實 ──
  const mn = h.minority_vs_naive || {};
  if (mn.reflexive && mn.reflexive.n && mn.naive && mn.naive.n) {
    await sleep(3200);
    const rf = mn.reflexive, nv = mn.naive;
    const popName = popular.name.slice(0, 10);
    if (rf.avg_reroutes < nv.avg_reroutes - 0.05) {
      // 反身者真的贏 → minority game 成立
      $("caption").className = "g-caption win";
      $("caption").innerHTML =
        `第三種人：少數「<b>預期羊群、反其道而行</b>」的駕駛，` +
        ` 個人平均繞圈 <b>${rf.avg_reroutes}</b> 次 ＜ 從眾者 <b>${nv.avg_reroutes}</b> 次。` +
        `　當大家都搶同一條街，偏離群眾的少數反而贏。（minority game）`;
    } else {
      // 誠實：本場熱門街容量夠大，從眾沒吃虧 → 反骨反而繞更多
      $("caption").className = "g-caption";
      $("caption").innerHTML =
        `但這場有個轉折：「${popName}」車位夠多，從眾其實停得下，` +
        ` 反而<b>反其道而行的人繞更多</b>（${rf.avg_reroutes} vs ${nv.avg_reroutes} 次）。` +
        `　羊群效應只在「熱門選擇真的不夠停」時才反咬 —— 這正是協調的價值所在。`;
    }
  }

  playing = false; $("btn-play").disabled = false;
}

function fillCell(containerId, segId, n, pilot) {
  const lot = $(`${containerId}-${segId}`);
  if (!lot) return;
  const cells = lot.querySelectorAll(".cell");
  const cell = cells[n - 1];
  if (cell) { cell.classList.add("filled"); if (pilot) cell.classList.add("pilot"); }
}

function animateCar(roadId, oc, pilot) {
  const road = $(roadId);
  const car = document.createElement("div");
  car.className = "car";
  car.textContent = oc.succeeded ? "🚗" : "🚙";
  const startX = Math.random() * (road.clientWidth - 40);
  car.style.left = startX + "px";
  road.appendChild(car);
  requestAnimationFrame(() => {
    if (oc.succeeded) {
      car.style.transform = `translateX(${(Math.random()*60-30)}px) translateY(-30px)`;
      setTimeout(() => car.classList.add("park"), 250);
    } else {
      car.classList.add("giveup");  // 開走離場
    }
  });
  setTimeout(() => car.remove(), 1200);
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

$("btn-run").onclick = runScenario;
$("btn-play").onclick = play;
$("btn-play").disabled = true;
// 自動載入一次
runScenario();
