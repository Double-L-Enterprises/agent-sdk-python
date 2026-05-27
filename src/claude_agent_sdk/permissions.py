"""Permission model for controlling tool access with approval callbacks."""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

# Type alias for approval callback
ApprovalCallback = Callable[[str, dict[str, Any]], Awaitable[bool]]


class PermissionPolicy:
    """Controls what tools agents can use, with approval callbacks."""
    
    def __init__(self) -> None:
        """Initialize a new permission policy."""
        self._allowed_tools: set[str] = set()
        self._denied_tools: set[str] = set()
        self._approval_callbacks: dict[str, ApprovalCallback] = {}
        self._sandbox_allowed_paths: list[str] | None = None
        self._sandbox_allow_network: bool = True
    
    def allow(self, *tool_names: str) -> PermissionPolicy:
        """Whitelist tools (chainable).
        
        Args:
            *tool_names: Tool names to allow
            
        Returns:
            Self for method chaining
        """
        self._allowed_tools.update(tool_names)
        return self
    
    def deny(self, *tool_names: str) -> PermissionPolicy:
        """Blacklist tools (chainable).
        
        Args:
            *tool_names: Tool names to deny
            
        Returns:
            Self for method chaining
        """
        self._denied_tools.update(tool_names)
        return self
    
    def require_approval(
        self, 
        *tool_names: str, 
        callback: ApprovalCallback
    ) -> PermissionPolicy:
        """Require approval for specific tools (chainable).
        
        Args:
            *tool_names: Tool names that require approval
            callback: Function called with (tool_name, params) that returns True to allow
            
        Returns:
            Self for method chaining
        """
        for tool_name in tool_names:
            self._approval_callbacks[tool_name] = callback
        return self
    
    def sandbox(
        self, 
        allowed_paths: list[str] | None = None, 
        allow_network: bool = True
    ) -> PermissionPolicy:
        """Restrict Bash and file tools to specific directories.
        
        Args:
            allowed_paths: List of allowed directory paths (None means all paths allowed)
            allow_network: Whether to allow network operations in Bash (curl/wget/pip install)
            
        Returns:
            Self for method chaining
        """
        self._sandbox_allowed_paths = allowed_paths
        self._sandbox_allow_network = allow_network
        return self
    
    async def check(self, tool_name: str, params: dict[str, Any]) -> tuple[bool, str]:
        """Check if a tool can be used with given parameters.
        
        Args:
            tool_name: Name of the tool to check
            params: Parameters that would be passed to the tool
            
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Check deny list first
        if tool_name in self._denied_tools:
            return False, f"Tool {tool_name} is denied by policy"
        
        # Check allow list
        if tool_name in self._allowed_tools:
            # If it's allowed but also requires approval, check approval
            if tool_name in self._approval_callbacks:
                try:
                    allowed = await self._approval_callbacks[tool_name](tool_name, params)
                    if not allowed:
                        return False, f"Tool {tool_name} requires approval but was denied"
                except Exception as e:
                    return False, f"Tool {tool_name} approval callback failed: {e}"
            
            # Check sandbox restrictions if applicable
            sandbox_result = self._check_sandbox(tool_name, params)
            if not sandbox_result[0]:
                return sandbox_result
            
            return True, "Tool allowed by policy"
        
        # If not explicitly allowed, check if it requires approval
        if tool_name in self._approval_callbacks:
            try:
                allowed = await self._approval_callbacks[tool_name](tool_name, params)
                if not allowed:
                    return False, f"Tool {tool_name} requires approval but was denied"
                
                # Check sandbox restrictions if approved
                sandbox_result = self._check_sandbox(tool_name, params)
                if not sandbox_result[0]:
                    return sandbox_result
                
                return True, "Tool approved by callback"
            except Exception as e:
                return False, f"Tool {tool_name} approval callback failed: {e}"
        
        # Default: deny if not explicitly allowed or approved
        return False, f"Tool {tool_name} is not allowed by policy"
    
    def _check_sandbox(self, tool_name: str, params: dict[str, Any]) -> tuple[bool, str]:
        """Check sandbox restrictions for file and network operations.
        
        Args:
            tool_name: Name of the tool
            params: Parameters for the tool
            
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Only apply sandbox restrictions to relevant tools
        file_tools = {"Read", "Write", "Edit", "Glob"}
        bash_tool = "Bash"
        
        if tool_name in file_tools:
            # Check file path restrictions
            if self._sandbox_allowed_paths is not None:
                file_path = params.get("file_path") or params.get("path")
                if file_path:
                    try:
                        resolved_path = Path(file_path).resolve()
                        allowed = False
                        for allowed_path in self._sandbox_allowed_paths:
                            allowed_path_resolved = Path(allowed_path).resolve()
                            if resolved_path.is_relative_to(allowed_path_resolved):
                                allowed = True
                                break
                        if not allowed:
                            return False, f"File path {file_path} is outside allowed sandbox paths"
                    except Exception as e:
                        return False, f"Error checking file path sandbox: {e}"
        
        elif tool_name == bash_tool:
            # Check network restrictions
            if not self._sandbox_allow_network:
                command = params.get("command", "")
                # Check for common network commands
                network_patterns = [
                    r'\bcurl\b', r'\bwget\b', r'\bpip\s+install\b',
                    r'\bgit\s+clone\b', r'\bgit\s+pull\b', r'\bgit\s+fetch\b'
                ]
                for pattern in network_patterns:
                    if re.search(pattern, command, re.IGNORECASE):
                        return False, f"Network command detected in Bash: {command}"
            
            # Check working directory restrictions
            if self._sandbox_allowed_paths is not None:
                cwd = params.get("cwd")
                if cwd:
                    try:
                        resolved_cwd = Path(cwd).resolve()
                        allowed = False
                        for allowed_path in self._sandbox_allowed_paths:
                            allowed_path_resolved = Path(allowed_path).resolve()
                            if resolved_cwd.is_relative_to(allowed_path_resolved):
                                allowed = True
                                break
                        if not allowed:
                            return False, f"Bash working directory {cwd} is outside allowed sandbox paths"
                    except Exception as e:
                        return False, f"Error checking Bash cwd sandbox: {e}"
        
        return True, "Sandbox check passed"
    
    def as_hook(self) -> Callable[..., Awaitable[tuple[bool, str]]]:
        """Return an async function compatible with HookRegistry for pre_tool_use hooks.
        
        Returns:
            Async function that can be used as a hook
        """
        async def hook_func(**kwargs: Any) -> tuple[bool, str]:
            tool_name = kwargs.get("tool_name")
            params = kwargs.get("params", {})
            if tool_name is None:
                return False, "No tool name provided to permission hook"
            return await self.check(tool_name, params)
        return hook_func