from plap.bus import bus
from plap.plugins.chat_completions.routes import create_chat_completion


@bus.listen("bootstrap.routes")
async def bootstrap_routes(routes: tuple[object, ...], *, next):
    return await next(routes=(*routes, create_chat_completion))


__all__ = ["bootstrap_routes", "create_chat_completion"]
