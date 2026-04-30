from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("Money Runtime MCP")


@mcp.tool(
    name="money_search",
    description="Search a deterministic local money-test corpus.",
)
def money_search(query: str) -> str:
    """Return a deterministic search result for runtime conformance tests."""
    return (
        "money_search_result marker=runtime-mcp-731 "
        f"query={query!r} source=local-money-mcp"
    )


if __name__ == "__main__":
    mcp.run()
