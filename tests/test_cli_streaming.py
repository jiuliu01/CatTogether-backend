import asyncio
import sys

from agents.base import InvokeContext
from agents.cli.cli_base import CLIBaseAgent


class LargeLineAgent(CLIBaseAgent):
    def __init__(self) -> None:
        super().__init__("large-line", "Large Line", sys.executable)

    def build_command(self, ctx: InvokeContext) -> list[str]:
        return [sys.executable, "-c", "print('x' * 70000)"]


def test_cli_accepts_stdout_line_larger_than_asyncio_default():
    async def collect():
        return [
            event
            async for event in LargeLineAgent().invoke(
                InvokeContext(channel_id="ch", user_message="test")
            )
        ]

    events = asyncio.run(collect())

    errors = [data["message"] for event_type, data in events if event_type == "error"]
    text = "".join(data["delta"] for event_type, data in events if event_type == "text_delta")

    assert errors == []
    assert len(text.strip()) == 70000
    assert events[-1] == ("done", {})
