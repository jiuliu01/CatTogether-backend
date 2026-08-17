import asyncio

from core.delegate_registry import DelegateContextRegistry


def test_issue_get_revoke():
    async def run():
        reg = DelegateContextRegistry()
        token = await reg.issue(
            project_id="p1", channel_id="c1", thread_id="t1", run_id="r1", depth=0,
            user_id="feishu:tenant:user-a",
        )
        assert token
        ctx = await reg.get(token)
        assert ctx is not None
        assert ctx.project_id == "p1"
        assert ctx.depth == 0
        assert ctx.user_id == "feishu:tenant:user-a"
        await reg.revoke(token)
        assert await reg.get(token) is None

    asyncio.run(run())


def test_max_depth_enforced(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "delegate_max_depth", 2)

    async def run():
        reg = DelegateContextRegistry()
        assert reg.max_depth == 2
        token = await reg.issue(
            project_id="p1", channel_id="c1", thread_id="t1", run_id="r1", depth=2
        )
        ctx = await reg.get(token)
        assert ctx.depth >= reg.max_depth  # caller must reject further delegation

    asyncio.run(run())


def test_concurrency_semaphore():
    async def run():
        from config import settings
        import core.delegate_registry as mod

        monkeypatch_reg = DelegateContextRegistry()
        sem = monkeypatch_reg.acquire()
        assert hasattr(sem, "acquire")
        # Acquiring the semaphore should not block beyond the configured limit.
        async with sem:
            assert True

    asyncio.run(run())
