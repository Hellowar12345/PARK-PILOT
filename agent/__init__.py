"""PARK-PILOT agentic AI + IoT 停車代理人 (模擬優先版)。"""
import os
from pathlib import Path


def _load_dotenv() -> None:
    """極簡 .env loader (避免多裝 python-dotenv)。"""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


_load_dotenv()

from .routes import router as agent_router  # noqa: E402
from .state import session  # noqa: E402
from .watcher import watcher_loop  # noqa: E402

__all__ = ["agent_router", "watcher_loop", "session"]
