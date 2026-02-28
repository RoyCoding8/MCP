"""ReasonForge — Code Expert MCP Server.

Run:   uv run python -m experts.code.server
Test:  uv run mcp dev experts/code/server.py
"""

from mcp.server.fastmcp import FastMCP
from experts.code.tools.code import code_tool
from experts.code.tools.text import text_tool

mcp = FastMCP(
    name="ReasonForge-Code",
    instructions=(
        "You are a precise coding assistant. You have access to 2 deterministic tools. "
        "Use code_tool to run, check, or inspect Python code. "
        "Use text_tool for regex, diffs, and text manipulation. "
        "ALWAYS use these tools — never guess output. Be concise."
    ),
)

# Register tools
mcp.tool()(code_tool)
mcp.tool()(text_tool)

if __name__ == "__main__":
    mcp.run()
