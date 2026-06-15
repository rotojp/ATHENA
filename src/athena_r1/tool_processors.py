"""Tool-call dispatch for ATHENA-R1.

Each ``<tool_call>...</tool_call>`` block emitted by the agent is routed to a
concrete ``ToolCallProcessor`` subclass. Special tools (``Tool_RAG``,
``CallAgent``, ``Finish``, ``DirectResponse``, ``RequireClarification``) have
dedicated processors; everything else falls through to ``DefaultToolProcessor``
which executes the named ToolUniverse function.
"""

import ast
import json
import logging

logger = logging.getLogger(__name__)


class ToolCallProcessor:
    """Base class for tool call processors."""

    def can_handle(self, tool_name: str) -> bool:
        """Check if this processor can handle the given tool."""
        raise NotImplementedError

    def process(self, tool_call: dict, context: dict) -> tuple[str, dict]:
        """Dispatch the tool call.

        Args:
            tool_call: Tool call dict with ``name`` and ``arguments`` keys.
            context: Per-turn context (conversation, callbacks, agent state).

        Returns:
            ``(call_result_string, context_updates_dict)``.
        """
        raise NotImplementedError


class FinishToolProcessor(ToolCallProcessor):
    """Processor for Finish tool calls."""

    def can_handle(self, tool_name: str) -> bool:
        return tool_name == "Finish"

    def process(self, tool_call: dict, context: dict) -> tuple[str, dict]:
        """Finish tool stops the conversation."""
        # The final result will be extracted from revised_messages[0]['content']
        # This is handled in run_function_call_stream
        return "", {"should_stop": True, "special_tool_call": "Finish"}


class DirectResponseToolProcessor(ToolCallProcessor):
    """Processor for DirectResponse tool calls."""

    def can_handle(self, tool_name: str) -> bool:
        return tool_name == "DirectResponse"

    def process(self, tool_call: dict, context: dict) -> tuple[str, dict]:
        """DirectResponse returns a direct response to the user."""
        # The model emits this argument misspelled as "respose"; keep that as
        # the primary key (it matches model output) with "response" as a
        # defensive fallback. Do not drop the misspelling.
        call_result = tool_call["arguments"].get("respose") or tool_call["arguments"].get(
            "response", ""
        )
        return call_result, {"should_stop": True, "special_tool_call": "DirectResponse"}


class RequireClarificationToolProcessor(ToolCallProcessor):
    """Processor for RequireClarification tool calls."""

    def can_handle(self, tool_name: str) -> bool:
        return tool_name == "RequireClarification"

    def process(self, tool_call: dict, context: dict) -> tuple[str, dict]:
        """RequireClarification asks user for clarification."""
        call_result = tool_call["arguments"].get("unclear_question", "")
        return call_result, {"should_stop": True, "special_tool_call": "RequireClarification"}


class ToolRAGProcessor(ToolCallProcessor):
    """Processor for Tool_RAG tool calls."""

    def __init__(self, agent_instance):
        """Initialize with agent instance to access tool_RAG method."""
        self.agent = agent_instance

    def can_handle(self, tool_name: str) -> bool:
        return tool_name == "Tool_RAG"

    def process(self, tool_call: dict, context: dict) -> tuple[str, dict]:
        """Tool_RAG retrieves additional tools by calling ToolUniverse directly."""
        progress_callback = context.get("progress_callback")
        agent_id = context.get("agent_id", "main")
        agent_level = context.get("call_agent_level", 0)

        # Emit tool_rag_query event so the UI can show "Searching tool library for: X".
        if progress_callback is not None:
            try:
                query_args = tool_call.get("arguments", {}) or {}
                query_text = (
                    query_args.get("description")
                    or query_args.get("query")
                    or json.dumps(query_args, ensure_ascii=False)
                )
                progress_callback(
                    {
                        "type": "tool_rag_query",
                        "content": query_text,
                        "agent_id": agent_id,
                        "agent_level": agent_level,
                        "arguments": query_args,
                    }
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"progress_callback raised in tool_rag_query: {e!r}")

        try:
            # Call ToolUniverse's Tool_RAG tool directly
            call_result = self.agent.tooluniverse.run_one_function(tool_call)

            # Check if result is an error
            if isinstance(call_result, dict) and "error" in call_result:
                error_msg = f"Tool_RAG failed: {call_result.get('error', 'Unknown error')}"
                logger.info(f"{error_msg}")
                return json.dumps({"error": error_msg}), {
                    "existing_tools_prompt": context.get("existing_tools_prompt", []),
                    "special_tool_call": "Tool_RAG",
                }

            # Tool_RAG returns a list of tool specifications directly
            # Parse the result
            if isinstance(call_result, str):
                try:
                    new_tools = json.loads(call_result)
                except (json.JSONDecodeError, TypeError):
                    # Try ast.literal_eval as a fallback if json.loads fails
                    try:
                        new_tools = ast.literal_eval(call_result)
                    except Exception:
                        new_tools = []
                        logger.info(f"Tool_RAG Result JSON/AST Error: {new_tools}")
            elif isinstance(call_result, list):
                new_tools = call_result
            else:
                new_tools = []

            # Convert tool specifications to prompt format
            existing_tools_prompt = context.get("existing_tools_prompt", [])
            if context.get("update_tool_list", True):
                if new_tools:
                    new_tools_prompt = self.agent.tooluniverse.prepare_tool_prompts(new_tools)

                    if isinstance(existing_tools_prompt, list):
                        existing_tools_prompt += new_tools_prompt
                    else:
                        existing_tools_prompt = new_tools_prompt
                else:
                    # If no tools found, keep existing tools prompt as is
                    if not isinstance(existing_tools_prompt, list):
                        existing_tools_prompt = []

            # Emit tools_retrieved so the UI can list what's now available.
            if progress_callback is not None:
                try:
                    tool_names = []
                    if isinstance(new_tools, list):
                        for t in new_tools:
                            if isinstance(t, dict):
                                name = t.get("name") or t.get("tool_name")
                                if name:
                                    tool_names.append(name)
                    progress_callback(
                        {
                            "type": "tools_retrieved",
                            "content": ", ".join(tool_names),
                            "agent_id": agent_id,
                            "agent_level": agent_level,
                            "tools": tool_names,
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"progress_callback raised in tools_retrieved: {e!r}")

            # Return full tool specifications as JSON string for UI display
            return json.dumps(new_tools, indent=1), {
                "existing_tools_prompt": existing_tools_prompt,
                "special_tool_call": "Tool_RAG",
            }
        except Exception as e:
            error_msg = f"Tool_RAG exception: {str(e)}"
            logger.info(f"{error_msg}")
            import traceback

            traceback.print_exc()
            return json.dumps({"error": error_msg}), {
                "existing_tools_prompt": context.get("existing_tools_prompt", []),
                "special_tool_call": "Tool_RAG",
            }


class CallAgentProcessor(ToolCallProcessor):
    """Processor for CallAgent tool calls — spawns a child multi-step agent.

    The child runs as a recursive `agent.run_multistep_agent` call. If a
    `progress_callback` is present on the parent context, child events are
    re-tagged with a fresh `agent_id` (e.g. ``main.sub-1``) and forwarded so
    streaming UIs can render the subagent inline.
    """

    def __init__(self, agent_instance):
        import threading as _threading

        self.agent = agent_instance
        self._subagent_counter: dict[str, int] = {}
        self._counter_lock = _threading.Lock()

    def can_handle(self, tool_name: str) -> bool:
        return tool_name == "CallAgent"

    def reset_counter(self) -> None:
        """Zero the sub-agent counter so the next top-level run gets sub-1
        again. Without this, the counter accumulates across every call on a
        long-running server (the registry is a singleton on the agent
        instance), producing user-visible agent_ids like ``main.sub-237``
        after enough traffic."""
        with self._counter_lock:
            self._subagent_counter.clear()

    def _next_subagent_id(self, parent_id: str) -> str:
        # Locked so concurrent CallAgent processors don't collide on the same
        # sibling index when the engine parallelizes a batch of CallAgents.
        with self._counter_lock:
            n = self._subagent_counter.get(parent_id, 0) + 1
            self._subagent_counter[parent_id] = n
        return f"{parent_id}.sub-{n}"

    def process(self, tool_call: dict, context: dict) -> tuple[str, dict]:
        call_agent_level = context.get("call_agent_level", 0)
        call_agent = context.get("call_agent", False)
        parent_agent_id = context.get("agent_id", "main")
        progress_callback = context.get("progress_callback")

        # Respect agent depth policy
        max_agent_level = getattr(self.agent, "max_agent_level", None)
        if (
            max_agent_level is not None
            and max_agent_level > 0
            and call_agent_level >= max_agent_level
        ):
            call_agent = False

        if not call_agent:
            return (
                "Error: The CallAgent has been disabled. Please proceed with "
                "your reasoning process to solve this question.",
                {"special_tool_call": "CallAgent"},
            )

        tool_args = tool_call.get("arguments", {}) or {}
        solution_plan = tool_args.get("solution") or tool_args.get("plan") or ""

        # Generate a unique child agent ID for nested streaming.
        child_id = self._next_subagent_id(parent_agent_id)
        next_level = call_agent_level + 1
        child_call_agent = True
        if max_agent_level is not None and next_level >= max_agent_level:
            child_call_agent = False

        # Emit subagent_start so the UI can open a nested swimlane.
        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        "type": "subagent_start",
                        "content": str(solution_plan),
                        "agent_id": child_id,
                        "parent_agent_id": parent_agent_id,
                        "agent_level": next_level,
                    }
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"progress_callback raised in subagent_start: {e!r}")

        # Wrap the parent's callback so child events are re-tagged with the
        # child's agent_id / agent_level / parent.
        if progress_callback is not None:

            def child_callback(event: dict) -> None:
                event.setdefault("agent_id", child_id)
                event.setdefault("parent_agent_id", parent_agent_id)
                event.setdefault("agent_level", next_level)
                progress_callback(event)
        else:
            child_callback = None

        # Recurse into a fresh multi-step run for the child task.
        try:
            child_response, _child_conv = self.agent.run_multistep_agent(
                message=str(solution_plan),
                temperature=context.get("temperature", 0.7),
                top_p=context.get("top_p", 0.95),
                top_k=context.get("top_k", 20),
                min_p=context.get("min_p", 0.0),
                max_new_tokens=context.get("max_new_tokens", 4096),
                max_round=getattr(self.agent, "child_max_round", 20),
                call_agent=child_call_agent,
                call_agent_level=next_level,
                return_conversation=True,
                progress_callback=child_callback,
                agent_id=child_id,
                # Propagate the user-abort signal — without this, sub-agents
                # keep running (and recursively spawning more) after the user
                # hits Stop on the top-level run.
                cancel_event=context.get("cancel_event"),
            )
            call_result = child_response or "Error: Child agent did not produce a result."
            if isinstance(call_result, str) and "</think>" in call_result:
                call_result = call_result.split("</think>")[-1].strip()
        except Exception as e:  # noqa: BLE001
            logger.info(f"CallAgent execution failed: {e}")
            call_result = f"Error in CallAgent: {e}"

        # Emit subagent_end (the parent will then incorporate this result).
        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        "type": "subagent_end",
                        "content": call_result,
                        "agent_id": child_id,
                        "parent_agent_id": parent_agent_id,
                        "agent_level": next_level,
                    }
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"progress_callback raised in subagent_end: {e!r}")

        callagent_plans = context.get("callagent_plans", [])
        callagent_plans.append(solution_plan)
        return call_result, {
            "callagent_plans": callagent_plans,
            "special_tool_call": "CallAgent",
        }


class DefaultToolProcessor(ToolCallProcessor):
    """Processor for default/normal tool calls."""

    def __init__(self, agent_instance):
        """Initialize with agent instance to access tooluniverse."""
        self.agent = agent_instance

    def can_handle(self, tool_name: str) -> bool:
        # Default processor handles all tools not handled by special processors
        return True

    def process(self, tool_call: dict, context: dict) -> tuple[str, dict]:
        """Process normal tool call using tooluniverse."""
        call_result = self.agent.tooluniverse.run_one_function(tool_call)
        return call_result, {}


class ToolProcessorRegistry:
    """Routes a tool name to the first processor that claims it."""

    def __init__(self, agent_instance):
        self.agent = agent_instance
        self.processors: list[ToolCallProcessor] = [
            FinishToolProcessor(),
            DirectResponseToolProcessor(),
            RequireClarificationToolProcessor(),
            ToolRAGProcessor(agent_instance),
            CallAgentProcessor(agent_instance),
            DefaultToolProcessor(agent_instance),  # Must be last (catch-all)
        ]

    def get_processor(self, tool_name: str) -> ToolCallProcessor:
        """Get the appropriate processor for a tool."""
        for processor in self.processors:
            if processor.can_handle(tool_name):
                return processor
        return self.processors[-1]  # Default processor
