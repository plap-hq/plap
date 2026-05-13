from __future__ import annotations

from plap.persistence import create_database_engine


async def test_create_database_engine_enables_pool_pre_ping() -> None:
    engine = create_database_engine("postgresql+asyncpg://user:pass@localhost/test")
    try:
        assert engine.sync_engine.pool._pre_ping is True
    finally:
        await engine.dispose()
