"""Official Feishu SDK long-connection lifecycle."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import threading
from typing import Any

from config import settings
from integrations.feishu.event_handler import FeishuEventHandler, feishu_event_handler

logger = logging.getLogger(__name__)


class FeishuLongConnectionClient:
    def __init__(self, handler: FeishuEventHandler | None = None) -> None:
        self.handler = handler or feishu_event_handler
        self._client: Any = None
        self._thread: threading.Thread | None = None
        self._sdk_loop: asyncio.AbstractEventLoop | None = None
        self._start_error: str | None = None
        self._stopping = False

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def connected(self) -> bool:
        return bool(self.running and self._client and getattr(self._client, "_conn", None))

    @property
    def start_error(self) -> str | None:
        return self._start_error

    def start(self, loop: asyncio.AbstractEventLoop) -> bool:
        if self.running:
            return True
        if not settings.feishu_enabled:
            logger.info("Feishu integration disabled: credentials are not configured")
            return False
        if settings.feishu_connection_mode != "long_connection":
            self._start_error = (
                f"unsupported FEISHU_CONNECTION_MODE: {settings.feishu_connection_mode}"
            )
            logger.error(self._start_error)
            return False
        if importlib.util.find_spec("lark_oapi") is None:
            self._start_error = "lark-oapi is not installed"
            logger.error(self._start_error)
            return False

        def run() -> None:
            sdk_loop = asyncio.new_event_loop()
            self._sdk_loop = sdk_loop
            asyncio.set_event_loop(sdk_loop)
            try:
                # lark-oapi stores an event loop in its ws.client module. The
                # SDK must therefore be imported and bound inside this thread,
                # not from Uvicorn's already-running event loop.
                import lark_oapi as lark
                lark.ws.client.loop = sdk_loop

                def on_message(data) -> None:
                    try:
                        raw = lark.JSON.marshal(data)
                        payload = json.loads(raw) if isinstance(raw, str) else raw
                        if isinstance(payload, dict):
                            self.handler.submit_threadsafe(payload, loop)
                    except Exception:
                        logger.exception("failed to decode Feishu SDK event")

                dispatcher = (
                    lark.EventDispatcherHandler.builder("", "")
                    .register_p2_im_message_receive_v1(on_message)
                    .build()
                )
                self._client = lark.ws.Client(
                    settings.feishu_app_id,
                    settings.feishu_app_secret,
                    event_handler=dispatcher,
                    log_level=lark.LogLevel.INFO,
                    domain=settings.feishu_api_base_url,
                )
                self._client.start()
            except Exception as exc:
                if not self._stopping:
                    self._start_error = str(exc)
                    logger.exception("Feishu long connection stopped unexpectedly")
            finally:
                try:
                    pending = asyncio.all_tasks(sdk_loop)
                    for task in pending:
                        task.cancel()
                    if pending and not sdk_loop.is_running():
                        sdk_loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                if not sdk_loop.is_closed():
                    sdk_loop.close()
                self._sdk_loop = None

        self._stopping = False
        self._start_error = None
        self._thread = threading.Thread(
            target=run,
            name="feishu-long-connection",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stopping = True
        sdk_loop = self._sdk_loop
        client = self._client
        if sdk_loop and sdk_loop.is_running():
            if client is not None:
                try:
                    client._auto_reconnect = False
                    future = asyncio.run_coroutine_threadsafe(
                        client._disconnect(), sdk_loop
                    )
                    future.result(timeout=2)
                except Exception:
                    logger.warning("failed to stop Feishu client cleanly")
            sdk_loop.call_soon_threadsafe(sdk_loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None
        self._client = None
        self._sdk_loop = None


feishu_client = FeishuLongConnectionClient()
