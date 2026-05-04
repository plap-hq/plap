from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_database_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url)


def create_session_maker(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


class Database:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._engines_by_loop: dict[asyncio.AbstractEventLoop, AsyncEngine] = {}
        self._session_makers_by_loop: dict[asyncio.AbstractEventLoop, async_sessionmaker[AsyncSession]] = {}

    def session_maker(self) -> async_sessionmaker[AsyncSession]:
        loop = asyncio.get_running_loop()
        session_maker = self._session_makers_by_loop.get(loop)
        if session_maker is not None:
            return session_maker

        engine = create_database_engine(self._database_url)
        session_maker = create_session_maker(engine)
        self._engines_by_loop[loop] = engine
        self._session_makers_by_loop[loop] = session_maker
        return session_maker

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_maker()() as session:
            yield session

    @asynccontextmanager
    async def session_transaction(self) -> AsyncIterator[AsyncSession]:
        async with self.session_maker().begin() as session:
            yield session

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AsyncConnection]:
        async with self.engine().connect() as connection:
            yield connection

    @asynccontextmanager
    async def connection_transaction(self) -> AsyncIterator[AsyncConnection]:
        async with self.engine().begin() as connection:
            yield connection

    def engine(self) -> AsyncEngine:
        loop = asyncio.get_running_loop()
        engine = self._engines_by_loop.get(loop)
        if engine is not None:
            return engine

        engine = create_database_engine(self._database_url)
        self._engines_by_loop[loop] = engine
        self._session_makers_by_loop[loop] = create_session_maker(engine)
        return engine

    async def dispose_all(self) -> None:
        for engine in self._engines_by_loop.values():
            await engine.dispose()
        self._engines_by_loop.clear()
        self._session_makers_by_loop.clear()
