"""MCP Protocol Handler for JSON-RPC 2.0 message handling.

This module provides the ProtocolHandler class that encapsulates:
- Tool registration and schema management
- JSON-RPC error code handling
- Capability negotiation during initialize
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from mcp import types
from mcp.server.lowlevel import Server

from src.observability.logger import get_logger


# JSON-RPC 2.0 Error Codes
class JSONRPCErrorCodes:
    """Standard JSON-RPC 2.0 error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


@dataclass
class ToolDefinition:
    """Definition of an MCP tool."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any]


@dataclass
class ProtocolHandler:
    """Handles MCP protocol operations including tool registration and execution.

    This class encapsulates:
    - Tool registration with schema validation
    - Tool execution with error handling
    - Capability declaration for initialize response

    Attributes:
        server_name: Name of the MCP server.
        server_version: Version string of the server.
        tools: Registry of available tools.
    """

    server_name: str
    server_version: str
    tools: Dict[str, ToolDefinition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize logger after dataclass initialization."""
        self._logger = get_logger(log_level="INFO")

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        if name in self.tools:
            raise ValueError(f"Tool '{name}' is already registered")
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )
        self._logger.info("Registered tool: %s", name)

    def get_tool_schemas(self) -> List[types.Tool]:
        return [
            types.Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.input_schema,
            )
            for tool in self.tools.values()
        ]

    async def execute_tool(
        self, name: str, arguments: Dict[str, Any]
    ) -> types.CallToolResult:
        if name not in self.tools:
            self._logger.warning("Tool not found: %s", name)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: Tool '{name}' not found")],
                isError=True,
            )

        tool = self.tools[name]
        try:
            self._logger.info("Executing tool: %s", name)
            result = await tool.handler(**arguments)
            if isinstance(result, types.CallToolResult):
                return result
            if isinstance(result, str):
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=result)],
                    isError=False,
                )
            if isinstance(result, list):
                return types.CallToolResult(content=result, isError=False)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(result))],
                isError=False,
            )
        except TypeError as e:
            self._logger.error("Invalid params for tool %s: %s", name, e)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: Invalid parameters - {e}")],
                isError=True,
            )
        except Exception as e:
            self._logger.exception("Internal error executing tool %s", name)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: Internal server error while executing '{name}'")],
                isError=True,
            )

    def get_capabilities(self) -> Dict[str, Any]:
        return {"tools": {} if self.tools else {}}


def _register_default_tools(protocol_handler: ProtocolHandler) -> None:
    from src.mcp_server.tools.query_knowledge_hub import register_tool as register_query_tool
    register_query_tool(protocol_handler)

    from src.mcp_server.tools.list_collections import register_tool as register_list_tool
    register_list_tool(protocol_handler)

    from src.mcp_server.tools.get_document_summary import register_tool as register_summary_tool
    register_summary_tool(protocol_handler)

    from src.mcp_server.tools.open_original_document import register_tool as register_open_doc_tool
    register_open_doc_tool(protocol_handler)

    from src.mcp_server.tools.query_market_data import register_tool as register_market_tool
    register_market_tool(protocol_handler)


def create_mcp_server(
    server_name: str,
    server_version: str,
    protocol_handler: Optional[ProtocolHandler] = None,
    register_tools: bool = True,
) -> Server:
    if protocol_handler is None:
        protocol_handler = ProtocolHandler(
            server_name=server_name,
            server_version=server_version,
        )

    if register_tools:
        _register_default_tools(protocol_handler)

    server = Server(server_name)

    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        return protocol_handler.get_tool_schemas()

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: Dict[str, Any]
    ) -> types.CallToolResult:
        return await protocol_handler.execute_tool(name, arguments)

    server._protocol_handler = protocol_handler  # type: ignore[attr-defined]
    return server


def get_protocol_handler(server: Server) -> ProtocolHandler:
    return server._protocol_handler  # type: ignore[attr-defined]
