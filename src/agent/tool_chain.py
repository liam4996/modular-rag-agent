"""Tool Chain Executor for multi-step tool workflows.

Supports chaining multiple tools together for complex tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

from src.agent.tool_caller import ToolRegistry, ToolResult
from src.core.settings import Settings


class ChainStepType(Enum):
    """Types of chain steps."""
    TOOL = "tool"           # Execute a tool
    TRANSFORM = "transform"  # Transform data
    CONDITION = "condition"  # Conditional branch
    PARALLEL = "parallel"   # Parallel execution


@dataclass
class ChainStep:
    """Single step in a tool chain."""
    name: str
    step_type: ChainStepType
    tool_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    transform: Optional[Callable[[Any], Dict[str, Any]]] = None
    condition: Optional[Callable[[Any], bool]] = None
    on_error: str = "stop"  # "stop", "skip", "fallback"


@dataclass
class ChainResult:
    """Result of executing a tool chain."""
    success: bool
    data: Any
    steps_executed: List[str] = field(default_factory=list)
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class ToolChainExecutor:
    """Executor for multi-step tool chains.
    
    Allows defining workflows that chain multiple tools together,
    with support for data transformation between steps.
    
    Example:
        >>> executor = ToolChainExecutor(settings)
        >>> 
        >>> # Define a chain: search -> summarize
        >>> chain = [
        ...     ChainStep(
        ...         name="search",
        ...         step_type=ChainStepType.TOOL,
        ...         tool_name="query_knowledge_hub",
        ...         params={"query": "RAG", "top_k": 5}
        ...     ),
        ...     ChainStep(
        ...         name="format_results",
        ...         step_type=ChainStepType.TRANSFORM,
        ...         transform=lambda data: {"formatted": str(data)}
        ...     )
        ... ]
        >>> 
        >>> result = executor.execute_chain(chain)
    """
    
    def __init__(self, settings: Settings):
        """Initialize tool chain executor.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.tool_registry = ToolRegistry(settings)
    
    def execute_chain(self, steps: List[ChainStep]) -> ChainResult:
        """Execute a chain of tools.
        
        Args:
            steps: List of chain steps to execute
            
        Returns:
            ChainResult with execution results
        """
        context: Dict[str, Any] = {}
        steps_executed: List[str] = []
        step_results: List[Dict[str, Any]] = []
        
        for step in steps:
            try:
                result = self._execute_step(step, context)
                
                if result["success"]:
                    context[step.name] = result["data"]
                    steps_executed.append(step.name)
                    step_results.append({
                        "step": step.name,
                        "success": True,
                        "data": result["data"]
                    })
                else:
                    # Handle error based on on_error strategy
                    if step.on_error == "stop":
                        return ChainResult(
                            success=False,
                            data=None,
                            steps_executed=steps_executed,
                            step_results=step_results,
                            error=f"Step '{step.name}' failed: {result.get('error')}"
                        )
                    elif step.on_error == "skip":
                        step_results.append({
                            "step": step.name,
                            "success": False,
                            "error": result.get("error"),
                            "skipped": True
                        })
                        continue
                    # "fallback" would try alternative
                    
            except Exception as e:
                return ChainResult(
                    success=False,
                    data=None,
                    steps_executed=steps_executed,
                    step_results=step_results,
                    error=f"Step '{step.name}' exception: {str(e)}"
                )
        
        # Return final context
        return ChainResult(
            success=True,
            data=context,
            steps_executed=steps_executed,
            step_results=step_results
        )
    
    def _execute_step(
        self, 
        step: ChainStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single chain step.
        
        Args:
            step: Chain step to execute
            context: Current execution context
            
        Returns:
            Step execution result
        """
        if step.step_type == ChainStepType.TOOL:
            return self._execute_tool_step(step, context)
        elif step.step_type == ChainStepType.TRANSFORM:
            return self._execute_transform_step(step, context)
        elif step.step_type == ChainStepType.CONDITION:
            return self._execute_condition_step(step, context)
        else:
            return {
                "success": False,
                "error": f"Unknown step type: {step.step_type}"
            }
    
    def _execute_tool_step(
        self, 
        step: ChainStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool step."""
        if not step.tool_name:
            return {
                "success": False,
                "error": "Tool step missing tool_name"
            }
        
        # Resolve parameters (support template variables)
        params = self._resolve_params(step.params, context)
        
        # Execute tool
        result = self.tool_registry.execute(step.tool_name, **params)
        
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error
        }
    
    def _execute_transform_step(
        self, 
        step: ChainStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a transform step."""
        if not step.transform:
            return {
                "success": False,
                "error": "Transform step missing transform function"
            }
        
        try:
            result = step.transform(context)
            return {
                "success": True,
                "data": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Transform failed: {str(e)}"
            }
    
    def _execute_condition_step(
        self, 
        step: ChainStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a condition step."""
        if not step.condition:
            return {
                "success": False,
                "error": "Condition step missing condition function"
            }
        
        try:
            result = step.condition(context)
            return {
                "success": True,
                "data": {"condition_result": result}
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Condition evaluation failed: {str(e)}"
            }
    
    def _resolve_params(
        self, 
        params: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve parameter templates using context.
        
        Supports simple template substitution like:
        {"query": "{previous_step.query}"}
        """
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and "{" in value and "}" in value:
                # Simple template substitution
                try:
                    resolved[key] = value.format(**context)
                except KeyError:
                    # If template variable not found, use original value
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved
    
    def create_search_chain(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[ChainStep]:
        """Create a standard search chain.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of chain steps
        """
        return [
            ChainStep(
                name="search",
                step_type=ChainStepType.TOOL,
                tool_name="query_knowledge_hub",
                params={"query": query, "top_k": top_k}
            ),
            ChainStep(
                name="format",
                step_type=ChainStepType.TRANSFORM,
                transform=lambda ctx: self._format_search_results(ctx.get("search", {}))
            )
        ]
    
    def _format_search_results(self, data: Any) -> Dict[str, Any]:
        """Format search results for display."""
        if not isinstance(data, dict):
            return {"formatted": str(data)}
        
        results = data.get("results", [])
        if not results:
            return {"formatted": "No results found", "count": 0}
        
        formatted = []
        for i, r in enumerate(results[:3], 1):
            content = r.get("content", "")[:150]
            score = r.get("score", 0)
            formatted.append(f"{i}. [{score:.3f}] {content}...")
        
        return {
            "formatted": "\n".join(formatted),
            "count": len(results),
            "has_more": len(results) > 3
        }
