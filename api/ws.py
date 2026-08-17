"""WebSocket endpoint: bridges the event bus to the frontend.

Downstream: subscribe to channel events and forward AgentEvent JSON.
Upstream: receive client commands (send_message / cancel / ping) and hand them
to the orchestrator. Two concurrent tasks per connection.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.event_bus import event_bus
from core.orchestrator import orchestrator
from core.session_manager import session_manager
from models.schemas import SendRequest

router = APIRouter()


@router.websocket("/ws/channels/{channel_id}")
async def channel_ws(ws: WebSocket, channel_id: str):
    await ws.accept()
    ch = await session_manager.get_channel(channel_id)
    if ch is None:
        await ws.send_json({"seq": 0, "channel_id": channel_id, "agent_id": "system",
                            "type": "error", "data": {"message": "channel not found"}})
        await ws.close()
        return

    stop = asyncio.Event()

    async def downstream():
        try:
            async for event in event_bus.subscribe(channel_id):
                if stop.is_set():
                    break
                await ws.send_json(event.model_dump(mode="json"))
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    async def upstream():
        try:
            while not stop.is_set():
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                action = msg.get("action")
                if action == "send_message":
                    req = SendRequest(
                        content=msg.get("content", ""),
                        mentioned_agents=msg.get("mentioned_agents"),
                        strategy=msg.get("strategy"),
                    )
                    try:
                        await orchestrator.dispatch(channel_id, req.content, req.mentioned_agents, req.strategy)
                    except Exception as e:
                        await event_bus.publish(channel_id, "system", "error", {"message": str(e)})
                elif action == "cancel":
                    orchestrator.cancel(channel_id)
                elif action == "ping":
                    await ws.send_json({"seq": 0, "channel_id": channel_id, "agent_id": "system",
                                        "type": "status", "data": {"state": "pong"}})
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            stop.set()

    dtask = asyncio.create_task(downstream())
    utask = asyncio.create_task(upstream())
    try:
        await asyncio.wait({dtask, utask}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop.set()
        for t in (dtask, utask):
            if not t.done():
                t.cancel()
