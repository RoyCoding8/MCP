"""ReasonForge — Math Expert MCP Server.

Run:   uv run python -m experts.math.server
Test:  uv run mcp dev experts/math/server.py
"""

from mcp.server.fastmcp import FastMCP
from experts.math.tools.algebra import math_tool
from experts.math.tools.calculus import calculus_tool
from experts.math.tools.matrix import matrix_tool
from experts.math.tools.statistics import statistics_tool

mcp = FastMCP(
    name="ReasonForge-Math",
    instructions=(
        "You are a precise mathematician. You have access to 4 deterministic math tools. "
        "ALWAYS use these tools for computation — never compute in your head. "
        "One tool call gives an exact, verified answer. Be concise."
    ),
)

# Register the 4 consolidated tools
mcp.tool()(math_tool)
mcp.tool()(calculus_tool)
mcp.tool()(matrix_tool)
mcp.tool()(statistics_tool)

if __name__ == "__main__":
    mcp.run()
