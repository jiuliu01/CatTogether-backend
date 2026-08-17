"""FastAPI application entry point.

Run: uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api.rest import router as rest_router
from api.ws import router as ws_router
from api.projects import router as projects_router
from api.internal import router as internal_router
from api.memory import router as memory_router, register_exception_handlers as register_memory_handlers
from bootstrap import register_builtins
from core.session_manager import session_manager
from core.task_queue import feishu_task_queue
from integrations.feishu.client import feishu_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    register_builtins()
    await session_manager.restore()
    try:
        await feishu_task_queue.start()
        await feishu_task_queue.restore_pending()
        feishu_client.start(asyncio.get_running_loop())
        yield
    finally:
        # shutdown: stop external intake before workers and persist sessions
        feishu_client.stop()
        await feishu_task_queue.stop()
        try:
            await session_manager.snapshot()
        except Exception:
            pass


app = FastAPI(title="CatTogether", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(rest_router)
app.include_router(ws_router)
app.include_router(projects_router)
app.include_router(internal_router)
app.include_router(memory_router)
register_memory_handlers(app)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "feishu": {
            "configured": settings.feishu_enabled,
            "connected": feishu_client.connected,
        },
    }
