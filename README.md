# 🅿️ PARK-PILOT

> **AI 驅動的智慧停車推薦 Agent**  
> 結合 Gemini Function Calling × LightGBM 佔用率預測，幫你在抵達前找到最佳停車位

---

## ✨ 專案簡介

PARK-PILOT 是一個以大型語言模型為核心的停車決策 Agent。使用者只需輸入目的地與預計抵達時間，Agent 會自動：

1. **地理解析** — 把口語地名（如「板橋車站」）轉換成座標
2. **周邊路段搜尋** — 找出目的地附近的候選停車路段
3. **佔用率預測** — 呼叫自訓練的 LightGBM 模型（R² = 0.96，32 特徵）預測各時間點的剩餘車位
4. **天氣感知** — 查詢抵達時的天氣，調整推薦策略
5. **即時回饋** — 整合 TDX 即時資料，若車位被佔滿則自動重新規劃

---

## 🏗️ 系統架構

```
PARK-PILOT/
├── agent/                  # AI Agent 核心
│   ├── orchestrator.py     # Gemini Function-Calling 主迴圈
│   ├── tools.py            # 工具派發器（geocode / predict / TDX）
│   ├── prompts.py          # System Prompt 建構 + Tools Schema
│   ├── simulator.py        # 停車場模擬器（Demo 模式）
│   ├── risk.py             # 風險分位數評估
│   ├── state.py            # 任務狀態管理
│   └── traces.py           # Agent 決策追蹤紀錄
├── demo/                   # Web Demo 應用
│   ├── app.py              # FastAPI 後端
│   ├── inference.py        # LightGBM 推論介面
│   └── static/             # 前端介面（HTML/CSS/JS）
│       ├── pilot.html      # 主要 Pilot 介面
│       ├── index.html      # 測試儀表板
│       └── game.html       # 停車模擬遊戲
├── .env.example            # 環境變數範本
├── pyproject.toml          # 套件設定
└── start.bat               # Windows 一鍵啟動
```

---

## 🤖 技術棧

| 層級 | 技術 |
|------|------|
| LLM Agent | Google Gemini (`gemini-2.0-flash-lite`) + Function Calling |
| 預測模型 | LightGBM（R² = 0.96，32 特徵，含時段/假日/歷史佔用率） |
| 即時資料 | TDX 交通資料平台（台灣路邊停車即時資訊） |
| 後端 | FastAPI + Server-Sent Events（SSE 串流） |
| 前端 | Vanilla HTML/CSS/JS |
| 套件管理 | [uv](https://github.com/astral-sh/uv) |

---

## 🚀 快速開始

### 前置需求

- Windows 10 / 11
- 網路連線（用於呼叫 Gemini API）
- [Gemini API Key](https://aistudio.google.com/app/apikey)（免費版即可）

### 安裝步驟

**1. 設定 API Key**

```bash
cp .env.example .env
```

打開 `.env`，填入你自己的 Gemini API Key：

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**2. 啟動伺服器**

雙擊執行 `start.bat`（第一次執行會自動安裝 uv、Python 與所有套件，約需 3–5 分鐘）

**3. 開啟瀏覽器**

| 頁面 | 網址 |
|------|------|
| 🤖 PARK-PILOT 主介面 | http://localhost:8000/pilot |
| 📊 測試儀表板 | http://localhost:8000/ |

---

## 🎮 使用方式

在 Pilot 介面中：

1. 選擇「**模擬模式**」（預設）
2. 輸入目的地，例如：`板橋車站，0.5~0.6 km 內`
3. 送出後，觀察 Agent 自動執行多輪工具呼叫（geocode → search → predict → recommend）
4. 若模擬停車位被佔滿，Agent 會自動「重新規劃」並通知

---

## ⚙️ 環境變數

| 變數 | 說明 | 必填 |
|------|------|------|
| `GEMINI_API_KEY` | Google Gemini API 金鑰 | ✅ |

> 請勿將 `.env` 檔案 commit 到版本控制，`.gitignore` 已設定排除。

---

## 📝 注意事項

- 預設 Port 為 `8000`，若被佔用請先關閉其他程式
- 看到 `Application startup complete` 代表伺服器啟動成功
- 第一次執行需等待環境安裝，之後啟動只需幾秒

---

## 📄 授權

MIT License
