"""
多智能体 RAG 系统 - 并行融合控制器

职责：
- 当 Router 决定并行检索时，同时调用多个 Agent
- 等待所有 Agent 完成
- 将结果全部写入 Blackboard
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from .state import AgentState
from .router_agent import AgentType


class ParallelFusionController:
    """
    并行融合控制器
    
    职责：
    - 当 Router 决定并行检索时，同时调用多个 Agent
    - 等待所有 Agent 完成
    - 将结果全部写入 Blackboard
    - 错误处理和日志记录
    
    使用场景：
    - HYBRID_SEARCH 意图：同时调用 SearchAgent 和 WebAgent
    - 复杂查询：需要多个数据源的查询
    """
    
    def __init__(self, search_func, web_func):
        """
        初始化并行融合控制器
        
        Args:
            search_func: Search Agent 的搜索函数
            web_func: Web Agent 的搜索函数
        """
        self.search_func = search_func
        self.web_func = web_func
    
    def execute_parallel_search(
        self,
        state: AgentState,
        agents_to_invoke: List[AgentType]
    ) -> AgentState:
        """
        并行执行多个 Agent
        
        Args:
            state: 共享状态
            agents_to_invoke: 要调用的 Agent 列表
        
        Returns:
            更新后的状态
        """
        futures = {}
        results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 提交所有 Agent 任务
            for agent_type in agents_to_invoke:
                if agent_type == AgentType.SEARCH:
                    future = executor.submit(
                        self._execute_search,
                        state.user_input,
                        state.conversation_history
                    )
                    futures[future] = "search"
                
                elif agent_type == AgentType.WEB:
                    # 可以读取本地搜索结果用于优化（如果有的话）
                    local_results = state.read_from_blackboard("local_results")
                    future = executor.submit(
                        self._execute_web,
                        state.user_input,
                        local_results
                    )
                    futures[future] = "web"
            
            # 等待所有任务完成
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    result = future.result()
                    results[agent_name] = {
                        "success": True,
                        "data": result
                    }
                except Exception as e:
                    results[agent_name] = {
                        "success": False,
                        "error": str(e)
                    }
        
        # 将结果写入 Blackboard
        for agent_name, result in results.items():
            if result["success"]:
                if agent_name == "search":
                    state.add_to_blackboard(
                        "local_results",
                        result["data"],
                        "parallel_controller"
                    )
                    state.add_metric("local_result_count", len(result["data"]))
                
                elif agent_name == "web":
                    state.add_to_blackboard(
                        "web_results",
                        result["data"],
                        "parallel_controller"
                    )
                    state.add_metric("web_result_count", len(result["data"]))
                
                state.add_execution_trace({
                    "agent": "parallel_controller",
                    "action": f"{agent_name}_completed",
                    "result_count": len(result["data"]),
                })
            else:
                state.add_execution_trace({
                    "agent": "parallel_controller",
                    "action": f"{agent_name}_failed",
                    "error": result["error"],
                })
        
        # 记录并行执行指标
        state.add_metric("parallel_execution", True)
        state.add_metric("agents_invoked", len(agents_to_invoke))
        
        return state
    
    def _execute_search(self, query: str, context: List[Dict]) -> List[Dict]:
        """
        执行本地搜索
        
        Args:
            query: 查询
            context: 对话历史
            
        Returns:
            搜索结果列表
        """
        try:
            results = self.search_func(query, context=context)
            return results if results else []
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")
    
    def _execute_web(self, query: str, local_results: List[Dict]) -> List[Dict]:
        """
        执行联网搜索
        
        Args:
            query: 查询
            local_results: 本地搜索结果（用于优化查询）
            
        Returns:
            联网搜索结果列表
        """
        try:
            # 如果有本地结果，可以优化查询
            if local_results:
                # 本地已有基础原理，搜索最新进展
                refined_query = f"{query} 2025 2026 最新进展"
            else:
                refined_query = query
            
            results = self.web_func(refined_query)
            return results if results else []
        except Exception as e:
            raise Exception(f"Web search failed: {str(e)}")
    
    def execute_sequential(
        self,
        state: AgentState,
        agents_to_invoke: List[AgentType]
    ) -> AgentState:
        """
        串行执行多个 Agent（备用方案）
        
        Args:
            state: 共享状态
            agents_to_invoke: 要调用的 Agent 列表
        
        Returns:
            更新后的状态
        """
        for agent_type in agents_to_invoke:
            if agent_type == AgentType.SEARCH:
                results = self._execute_search(
                    state.user_input,
                    state.conversation_history
                )
                state.add_to_blackboard("local_results", results, "sequential_controller")
                state.add_metric("local_result_count", len(results))
            
            elif agent_type == AgentType.WEB:
                local_results = state.read_from_blackboard("local_results")
                results = self._execute_web(state.user_input, local_results)
                state.add_to_blackboard("web_results", results, "sequential_controller")
                state.add_metric("web_result_count", len(results))
        
        state.add_metric("parallel_execution", False)
        return state
