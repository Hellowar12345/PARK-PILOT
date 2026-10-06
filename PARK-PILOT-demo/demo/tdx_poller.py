"""
TDX 即時路邊停車資料 poller。
在 FastAPI lifespan 內作為 asyncio 背景 task 啟動。

需要環境變數：
  TDX_CLIENT_ID      — TDX 平台 client_id
  TDX_CLIENT_SECRET  — TDX 平台 client_secret

TDX 申請（免費）：https://tdx.transportdata.tw/register
"""
from __future__ import annotations
import os
import asyncio
import logging
from datetime import datetime
import httpx

logger = logging.getLogger("tdx_poller")

TDX_AUTH_URL = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
# 新北市路邊停車即時剩餘車位（OffStreet API 含路邊 PBS）
TDX_PBS_URL  = "https://tdx.transportdata.tw/api/basic/v1/Parking/OffStreet/ParkingAvailability/City/NewTaipei?%24format=JSON"

POLL_INTERVAL = 60  # 秒


async def _get_token(client: httpx.AsyncClient) -> str:
    client_id     = os.getenv("TDX_CLIENT_ID", "")
    client_secret = os.getenv("TDX_CLIENT_SECRET", "")
    if not client_id:
        raise RuntimeError("TDX_CLIENT_ID 未設定")
    resp = await client.post(TDX_AUTH_URL, data={
        "grant_type": "client_credentials",
        "client_id":     client_id,
        "client_secret": client_secret,
    })
    resp.raise_for_status()
    return resp.json()["access_token"]


async def run_poller(live_source):
    """背景無窮迴圈：每 POLL_INTERVAL 秒更新一次 buffer"""
    if not os.getenv("TDX_CLIENT_ID"):
        logger.warning("TDX_CLIENT_ID 未設定，Live 模式 poller 不啟動")
        return

    logger.info("TDX poller 啟動")
    token = None
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            try:
                if token is None:
                    token = await _get_token(client)

                resp = await client.get(TDX_PBS_URL,
                                        headers={"Authorization": f"Bearer {token}"})
                if resp.status_code == 401:
                    token = await _get_token(client)
                    resp = await client.get(TDX_PBS_URL,
                                            headers={"Authorization": f"Bearer {token}"})
                resp.raise_for_status()

                data = resp.json()
                now_str = datetime.now().isoformat(timespec="seconds")
                count = 0
                for item in data:
                    seg_id = str(item.get("ParkingSegmentID", "")).strip()
                    if not seg_id:
                        continue
                    available = item.get("AvailableSpaces")
                    total     = item.get("TotalSpaces") or 0
                    if available is None or total <= 0:
                        continue
                    occ = 1.0 - available / total
                    live_source.push(seg_id, now_str, occ, available)
                    count += 1

                logger.info(f"TDX poll: {count} 路段更新")

            except Exception as e:
                logger.warning(f"TDX poll 失敗: {e}")
                token = None  # 下次重新取 token

            await asyncio.sleep(POLL_INTERVAL)
