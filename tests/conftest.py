import pytest

pytest_plugins = [
    "tests.pytest_plugins.database",
    "tests.pytest_plugins.server",
]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
