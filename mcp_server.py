# mcp_server.py
from __future__ import annotations
from typing import Dict, Any
import signal
import sys

from mcp.server.fastmcp import FastMCP
from src.rag_core import init_rag, answer_query_json

mcp = FastMCP("policy-rag")

@mcp.tool()
def rag_query(question: str, top_k: int = 3, max_refs: int = 1) -> Dict[str, Any]:
    init_rag()
    return answer_query_json(question, top_k=top_k, max_refs=max_refs)

def _shutdown(*_):
    print("Shutting down MCP server...", flush=True)
    sys.exit(0)

def main() -> None:
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    print("MCP server running (stdio). Waiting for client...", flush=True)
    mcp.run()

if __name__ == "__main__":
    main()

