"""ReasonForge — Code Expert MCP Server."""

from mcp.server.fastmcp import FastMCP
from experts.code.tools.code import code_tool

mcp = FastMCP(
    name="ReasonForge-Code",
    instructions=(
        "You are a precise coding assistant. "
        "Use code_tool to run, check, or inspect Python code. "
        "ALWAYS use code_tool — never guess output. Be concise."
    ),
)

mcp.tool()(code_tool)

if __name__ == "__main__":
    mcp.run()
