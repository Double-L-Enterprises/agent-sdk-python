"""MCP (Model Context Protocol) Tool Bridge implementation.

This module allows the runner to use MCP servers as tool providers,
supporting both stdio and SSE (Server-Sent Events) transport protocols.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    
    type: str  # "stdio" | "sse"
    command: Optional[str] = None  # For stdio - e.g., "npx", "python3"
    args: Optional[List[str]] = None  # For stdio - e.g., ["-m", "mcp_server"]
    url: Optional[str] = None  # For sse - e.g., "http://localhost:8030/sse"
    env: Optional[Dict[str, str]] = None  # Extra environment variables


class MCPBridge:
    """Bridge to MCP (Model Context Protocol) servers for tool provision.
    
    Supports both stdio (subprocess) and SSE (HTTP) transport protocols.
    """
    
    def __init__(self, servers: Dict[str, MCPServerConfig]):
        """Initialize the MCP bridge with server configurations.
        
        Args:
            servers: Dictionary mapping server names to their configurations.
        """
        self.servers = servers
        self._processes: Dict[str, subprocess.Process] = {}
        self._http_clients: Dict[str, httpx.AsyncClient] = {}
        self._next_id = 1
        
    async def connect(self) -> None:
        """Connect to all configured MCP servers.
        
        For stdio servers: starts subprocesses.
        For SSE servers: initializes HTTP clients.
        """
        for server_name, config in self.servers.items():
            if config.type == "stdio":
                if not config.command or not config.args:
                    raise ValueError(f"Stdio server {server_name} requires command and args")
                
                # Start subprocess
                env = {**subprocess.os.environ, **(config.env or {})}
                process = await asyncio.create_subprocess_exec(
                    config.command,
                    *config.args,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )
                self._processes[server_name] = process
                
            elif config.type == "sse":
                if not config.url:
                    raise ValueError(f"SSE server {server_name} requires url")
                
                # Initialize HTTP client
                client = httpx.AsyncClient()
                self._http_clients[server_name] = client
                
            else:
                raise ValueError(f"Unknown server type: {config.type}")
    
    async def _send_jsonrpc_stdio(self, server_name: str, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send JSON-RPC request to stdio server and receive response."""
        process = self._processes.get(server_name)
        if not process:
            raise RuntimeError(f"Stdio server {server_name} not connected")
        
        request_id = self._next_id
        self._next_id += 1
        
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id
        }
        if params is not None:
            request["params"] = params
            
        # Send request
        request_json = json.dumps(request) + "\n"
        process.stdin.write(request_json.encode())
        await process.stdin.drain()
        
        # Read response
        response_line = await process.stdout.readline()
        if not response_line:
            raise RuntimeError(f"No response from server {server_name}")
            
        response = json.loads(response_line.decode().strip())
        
        # Verify response ID matches
        if response.get("id") != request_id:
            raise RuntimeError(f"Response ID mismatch: expected {request_id}, got {response.get('id')}")
            
        return response
    
    async def _send_jsonrpc_sse(self, server_name: str, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send JSON-RPC request to SSE server and receive response."""
        client = self._http_clients.get(server_name)
        if not client:
            raise RuntimeError(f"SSE server {server_name} not connected")
            
        config = self.servers[server_name]
        if not config.url:
            raise RuntimeError(f"SSE server {server_name} has no URL configured")
            
        request_id = self._next_id
        self._next_id += 1
        
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id
        }
        if params is not None:
            request["params"] = params
            
        response = await client.post(config.url, json=request)
        response.raise_for_status()
        
        return response.json()
    
    async def discover_tools(self) -> List[Dict[str, Any]]:
        """Discover tools from all connected MCP servers.
        
        Returns:
            List of tool definitions in OpenAI function calling format.
        """
        all_tools = []
        
        for server_name, config in self.servers.items():
            try:
                if config.type == "stdio":
                    response = await self._send_jsonrpc_stdio(server_name, "tools/list")
                elif config.type == "sse":
                    response = await self._send_jsonrpc_sse(server_name, "tools/list")
                else:
                    continue
                    
                if "result" in response and "tools" in response["result"]:
                    mcp_tools = response["result"]["tools"]
                    # Convert MCP tools to OpenAI format
                    openai_tools = self._convert_mcp_to_openai(mcp_tools, server_name)
                    all_tools.extend(openai_tools)
                    
            except Exception as exc:
                logger.warning("Failed to discover tools from server %s: %s", server_name, exc)
                continue
                
        return all_tools
    
    def _convert_mcp_to_openai(self, mcp_tools: List[Dict[str, Any]], server_name: str) -> List[Dict[str, Any]]:
        """Convert MCP tool definitions to OpenAI function calling format.
        
        Args:
            mcp_tools: List of MCP tool definitions.
            server_name: Name of the server providing these tools.
            
        Returns:
            List of tool definitions in OpenAI format.
        """
        openai_tools = []
        
        for mcp_tool in mcp_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": f"{server_name}:{mcp_tool['name']}",
                    "description": mcp_tool.get("description", ""),
                    "parameters": mcp_tool.get("inputSchema", {})
                }
            }
            openai_tools.append(openai_tool)
            
        return openai_tools
    
    async def call_tool(self, server_name: str, tool_name: str, params: Dict[str, Any]) -> str:
        """Call a tool on a specific MCP server.
        
        Args:
            server_name: Name of the server to call.
            tool_name: Name of the tool to call (without server prefix).
            params: Tool parameters.
            
        Returns:
            Tool result as a string.
        """
        config = self.servers.get(server_name)
        if not config:
            raise ValueError(f"Unknown server: {server_name}")
            
        try:
            tool_call_params = {
                "name": tool_name,
                "arguments": params
            }
            
            if config.type == "stdio":
                response = await self._send_jsonrpc_stdio(server_name, "tools/call", tool_call_params)
            elif config.type == "sse":
                response = await self._send_jsonrpc_sse(server_name, "tools/call", tool_call_params)
            else:
                raise ValueError(f"Unknown server type: {config.type}")
                
            if "result" in response:
                result = response["result"]
                # Convert result to string if it's not already
                if isinstance(result, dict):
                    return json.dumps(result)
                elif isinstance(result, (list, tuple)):
                    return json.dumps(result)
                else:
                    return str(result)
            elif "error" in response:
                error = response["error"]
                return f"[ERROR] MCP tool call failed: {error.get('message', 'Unknown error')}"
            else:
                return "[ERROR] Invalid response from MCP server"
                
        except Exception as exc:
            logger.error("Tool call failed for %s:%s: %s", server_name, tool_name, exc)
            return f"[ERROR] Tool call failed: {exc}"
    
    async def close(self) -> None:
        """Close all connections to MCP servers.
        
        Terminates stdio subprocesses and closes HTTP clients.
        """
        # Close stdio processes
        for server_name, process in self._processes.items():
            try:
                process.terminate()
                await process.wait()
            except Exception as exc:
                logger.warning("Error terminating process for %s: %s", server_name, exc)
                
        self._processes.clear()
        
        # Close HTTP clients
        for server_name, client in self._http_clients.items():
            try:
                await client.aclose()
            except Exception as exc:
                logger.warning("Error closing HTTP client for %s: %s", server_name, exc)
                
        self._http_clients.clear()