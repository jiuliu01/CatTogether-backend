import asyncio
import sys

from agents.base import InvokeContext
from agents.cli.cli_base import CLIBaseAgent


class HangingAgent(CLIBaseAgent):
    def __init__(self) -> None:
        super().__init__("hanging", "Hanging", sys.executable)

    def build_command(self, ctx: InvokeContext) -> list[str]:
        # Sleep far longer than the per-invoke timeout; emit nothing on stdout.
        return [sys.executable, "-c", "import time; time.sleep(30)"]


def test_cli_invoke_times_out_and_reports_error():
    async def collect():
        return [
            event
            async for event in HangingAgent().invoke(
                InvokeContext(channel_id="ch", user_message="test", timeout=1.0)
            )
        ]

    events = asyncio.run(collect())

    types = [event_type for event_type, _ in events]
    errors = [data["message"] for event_type, data in events if event_type == "error"]

    assert "error" in types
    assert "done" not in types
    assert any("超时" in message for message in errors)
