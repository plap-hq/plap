from __future__ import annotations

import anyio
import pytest

from plap.persistence import create_database_engine
from plap.persistence.db import _shielded_exit


async def test_create_database_engine_enables_pool_pre_ping() -> None:
    engine = create_database_engine("postgresql+asyncpg://user:pass@localhost/test")
    try:
        assert engine.sync_engine.pool._pre_ping is True
    finally:
        await engine.dispose()


class _BlockingAsyncContextManager:
    def __init__(self) -> None:
        self.exit_started = anyio.Event()
        self.release_exit = anyio.Event()
        self.exit_calls: list[tuple[object, object, object]] = []

    async def __aenter__(self) -> str:
        return "resource"

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        self.exit_calls.append((exc_type, exc, traceback))
        self.exit_started.set()
        await self.release_exit.wait()
        return False


async def test_shielded_exit_runs_cleanup_under_cancellation() -> None:
    context_manager = _BlockingAsyncContextManager()
    cancelled_exc = anyio.get_cancelled_exc_class()
    result: dict[str, BaseException] = {}
    done = anyio.Event()

    async def runner() -> None:
        try:
            with anyio.CancelScope() as cancel_scope:
                async with _shielded_exit(context_manager):
                    cancel_scope.cancel()
                    await anyio.sleep(0)
        except BaseException as exc:
            result["error"] = exc
        finally:
            done.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(runner)
        await context_manager.exit_started.wait()
        context_manager.release_exit.set()
        await done.wait()

    assert result == {}
    assert context_manager.exit_calls
    exc_type, exc, _ = context_manager.exit_calls[0]
    assert exc_type is cancelled_exc
    assert isinstance(exc, cancelled_exc)


async def test_shielded_exit_preserves_body_exception_info() -> None:
    context_manager = _BlockingAsyncContextManager()
    context_manager.release_exit.set()

    with pytest.raises(RuntimeError, match="boom"):
        async with _shielded_exit(context_manager):
            raise RuntimeError("boom")

    exc_type, exc, _ = context_manager.exit_calls[0]
    assert exc_type is RuntimeError
    assert isinstance(exc, RuntimeError)
    assert str(exc) == "boom"
