
import os
import json
import time
from urllib import request, parse
from typing import Optional

def _truthy(val) -> bool:
    return str(val).strip().lower() in ("1","true","yes","on")

class TelegramNotifier:
    def __init__(self,
                 enabled: Optional[bool]=None,
                 token: Optional[str]=None,
                 chat_id: Optional[str]=None,
                 parse_mode: Optional[str]=None,
                 timeout: float = 5.0):
        env_enabled = os.getenv('TELEGRAM_ENABLED','false')
        self.enabled = _truthy(enabled) if enabled is not None else _truthy(env_enabled)
        self.token = (token or os.getenv('TELEGRAM_BOT_TOKEN','')).strip()
        self.chat_id = (chat_id or os.getenv('TELEGRAM_CHAT_ID','')).strip()
        self.parse_mode = (parse_mode or os.getenv('TELEGRAM_PARSE_MODE','HTML')).strip()
        self.timeout = timeout
        # anti-spam
        self._last_msg = ""
        self._last_ts = 0.0

    def _endpoint(self)->str:
        return f"https://api.telegram.org/bot{self.token}/sendMessage"

    def reason_not_ready(self)->str:
        if not self.enabled:
            return "TELEGRAM_ENABLED is not truthy (use true/1/yes/on)"
        if not self.token:
            return "TELEGRAM_BOT_TOKEN missing"
        if not self.chat_id:
            return "TELEGRAM_CHAT_ID missing"
        return ""

    def ready(self)->bool:
        return self.enabled and bool(self.token) and bool(self.chat_id)

    def send(self, text: str, disable_preview: bool=True, disable_notification: bool=False)->bool:
        if not self.ready():
            return False
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": disable_preview,
            "disable_notification": disable_notification,
        }
        if self.parse_mode:
            payload["parse_mode"] = self.parse_mode

        data = parse.urlencode(payload).encode("utf-8")
        req = request.Request(self._endpoint(), data=data, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    return False
                out = json.loads(resp.read().decode("utf-8"))
                return bool(out.get("ok"))
        except Exception:
            return False

    def send_throttled(self, text: str, min_gap_sec: float = 1.0)->bool:
        now = time.time()
        if text == self._last_msg and (now - self._last_ts) < min_gap_sec:
            return False
        ok = self.send(text)
        if ok:
            self._last_msg = text
            self._last_ts = now
        return ok
