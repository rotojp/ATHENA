import copy
import gc
import json
import logging
import os
import re
import threading
import traceback
from typing import Any, Dict, List, Optional

import httpx
from jinja2 import Template
from openai import AzureOpenAI, OpenAI
from tooluniverse import ToolUniverse
from transformers import AutoTokenizer

from ._prompts import (
    CONTEXT_SUMMARY_PROMPT,
    GPT_PLANNING_PROMPT,
    PROMPT_MULTI_STEP,
)
from .tool_processors import ToolProcessorRegistry
from .utils import convert_to_qwen_format

logger = logging.getLogger(__name__)


class AgentRunContext:
    def __init__(self, agent, message: str, call_agent: bool = False,
                 call_agent_level: int = 0, progress_callback=None,
                 agent_id: str = "main"):
        self.agent = agent
        self.message = message
        self.call_agent = call_agent
        self.call_agent_level = call_agent_level
        self.conversation = []
        self.picked_tools_prompt = []
        self.last_outputs = []
        self.last_status = {}
        self.token_overflow = False
        # Optional event sink: callable(event_dict). See progress events emitted
        # below for the schema. Used by AthenaR1.answer_streaming().
        self.progress_callback = progress_callback
        self.agent_id = agent_id

    def _emit(self, event_type: str, **fields):
        if self.progress_callback is None:
            return
        # Auto-tag every event with the current agent's id/level for the UI.
        fields.setdefault("agent_id", self.agent_id)
        fields.setdefault("agent_level", self.call_agent_level)
        try:
            self.progress_callback({"type": event_type, **fields})
        except Exception as e:  # noqa: BLE001
            logger.debug(f"progress_callback raised {e!r} — ignoring")

    def __enter__(self):
        logger.info("start")
        self.picked_tools_prompt, self.call_agent_level = self.agent.initialize_tools_prompt(
            self.call_agent, self.call_agent_level, self.message)
        self.conversation = self.agent.initialize_conversation(self.message, None, self.call_agent_level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.info(f"Agent run terminated with error: {exc_val}")
        return False

    def process_tool_calls(self, temperature, max_new_tokens, top_p, top_k, min_p, return_conversation):
        if len(self.last_outputs) > 0:
            result = self.agent.run_function_call(
                self.last_outputs[-1],
                existing_tools_prompt=self.picked_tools_prompt,
                message_for_call_agent=self.message,
                call_agent=self.call_agent,
                call_agent_level=self.call_agent_level,
                temperature=temperature,
                conversation=self.conversation,
                enable_summary=self.agent.enable_summary,
                last_status=self.last_status,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                progress_callback=self.progress_callback,
                agent_id=self.agent_id,
                cancel_event=getattr(self, "cancel_event", None))

            # Extract revised messages resulting from the tool call directly to conversation
            self.conversation.extend(result['messages'])
            self.picked_tools_prompt = result['tools_prompt']

            if result['should_stop']:
                return True, (result['final_result'], self.conversation) if return_conversation else result['final_result']
        return False, None

    def _condense_massive_tools(self, temperature, top_p, top_k, min_p):
        """Condenses individual tool outputs that exceed 30% of token capacity."""
        summarized_count = 0
        threshold = int(self.agent.max_token * self.agent.compress_ratio * 0.3)
        
        for i in range(len(self.conversation)):
            msg = self.conversation[i]
            if msg.get('role') == 'tool':
                content = msg.get('content', '')
                if not str(content).startswith("[Summarized Tool Output]:") and len(content) > threshold:
                    logger.info(f"Summarizing massive tool output at index {i} (len={len(content)})...")
                    thought_context = ""
                    if i > 0 and self.conversation[i-1].get('role') == 'assistant':
                        thought_context = self.conversation[i-1].get('content', '')

                    result_summary = self.agent.run_summary_agent(
                        thought_calls=thought_context,
                        function_response=content,
                        temperature=0.1,
                        max_new_tokens=1024,
                        top_p=top_p, top_k=top_k, min_p=min_p
                    )

                    if result_summary and len(result_summary) < len(content):
                        self.conversation[i]['full_content'] = content 
                        self.conversation[i]['content'] = f"[Summarized Tool Output]: {result_summary}"
                        summarized_count += 1
                        
        if summarized_count > 0:
            logger.info(f"First Pass Compaction completed. Summarized {summarized_count} massive tool results.")

    def _iteratively_compress_tools(self, safe_limit, temperature, top_p, top_k, min_p):
        """Loops over remaining unsummarized tools, largest first, until safe limit is met.

        Returns ``True`` iff ``safe_limit`` was reached. Returns ``False`` on
        stall (less than 1% progress) or exhaustion (no tools left to shrink).
        Skips small tool outputs (< 1000 chars) since summarizing them saves
        negligible space. The explicit boolean lets ``ensure_context_space``
        distinguish a successful Strategy-1 compaction from an exhausted one;
        without it every compaction returned ``None`` and Strategy 1 was
        silently skipped, forcing Strategy 2 on every overflow.
        """
        MIN_TOOL_SIZE = 1000
        prev_estimated_tokens = None
        while True:
            current_length_chars = sum(len(str(msg.get('content', ''))) for msg in self.conversation)
            tool_desc_chars = len(str(self.picked_tools_prompt))
            estimated_tokens = (current_length_chars + tool_desc_chars) / self.agent.compress_ratio
            
            if estimated_tokens <= safe_limit:
                return True

            # Bail out if we made less than 1% progress since last iteration
            if prev_estimated_tokens is not None:
                progress = prev_estimated_tokens - estimated_tokens
                if progress < prev_estimated_tokens * 0.01:
                    logger.info(f"Iterative compression stalled (saved only {int(progress)} tokens). Stopping.")
                    return False
            prev_estimated_tokens = estimated_tokens
            logger.info(f"Iterative Space Check: Context still at {int(estimated_tokens)} tokens. Condensing remaining tools.")
            
            unsummarized_tools = []
            for i, msg in enumerate(self.conversation):
                if msg.get('role') == 'tool':
                    content = msg.get('content', '')
                    if not str(content).startswith("[Summarized Tool Output]:") and len(content) >= MIN_TOOL_SIZE:
                        unsummarized_tools.append((i, len(str(content))))
            
            if not unsummarized_tools:
                logger.info("No tool outputs large enough to compress (all < 1000 chars). Stopping.")
                return False
                
            unsummarized_tools.sort(key=lambda x: x[1], reverse=True)
            target_idx = unsummarized_tools[0][0]
            
            content = self.conversation[target_idx].get('content', '')
            logger.info(f"Summarizing current largest tool output at index {target_idx} (len={len(content)})...")
            
            thought_context = ""
            if target_idx > 0 and self.conversation[target_idx-1].get('role') == 'assistant':
                thought_context = self.conversation[target_idx-1].get('content', '')

            result_summary = self.agent.run_summary_agent(
                thought_calls=thought_context,
                function_response=content,
                temperature=0.1,
                max_new_tokens=1024,
                top_p=top_p, top_k=top_k, min_p=min_p
            )

            if result_summary and len(result_summary) < len(content):
                self.conversation[target_idx]['full_content'] = content 
                self.conversation[target_idx]['content'] = f"[Summarized Tool Output]: {result_summary}"
            else:
                logger.info(f"Summarization failed to shrink content at index {target_idx}. Marking to skip.")
                self.conversation[target_idx]['content'] = f"[Summarized Tool Output]: {content}"

    def ensure_context_space(self, temperature, max_new_tokens, top_p, top_k, min_p, return_conversation, dynamic_limit=0.7, compact=False):
        """Preemptive Context Compaction (Strategy 1). Evaluates history and shrinks large tools."""
        if compact:
            logger.info(" Forced Strategy 2 Triggered. Bypassing tool shrink and checkpointing context... ")
            self._handle_token_overflow(temperature, max_new_tokens, top_p, top_k, min_p, return_conversation)
            return True

        safe_limit = self.agent.max_token * dynamic_limit
        
        current_length_chars = sum(len(str(msg.get('content', ''))) for msg in self.conversation)
        tool_desc_chars = len(str(self.picked_tools_prompt))
        estimated_tokens = (current_length_chars + tool_desc_chars) / self.agent.compress_ratio
        
        if estimated_tokens > safe_limit:
            logger.info(f"Preemptive Space Check: Context at {int(estimated_tokens)} tokens (Limit: {safe_limit} safe). Triggering Compaction.")
            
            # Step 1: Condense huge anomalies
            self._condense_massive_tools(temperature, top_p, top_k, min_p)
            
            # Step 2: Iteratively meet requirement
            success = self._iteratively_compress_tools(safe_limit, temperature, top_p, top_k, min_p)
            if not success:
                logger.info(" ensure_context_space exhausted all tools but failed to reach limit. ")
                return False

        # Final readout
        current_length_chars = sum(len(str(msg.get('content', ''))) for msg in self.conversation)
        tool_desc_chars = len(str(self.picked_tools_prompt))
        final_estimated_tokens = (current_length_chars + tool_desc_chars) / self.agent.compress_ratio
        
        if final_estimated_tokens <= safe_limit:
            logger.info(f" Space requirement met! Context fits well within safe limits ({int(final_estimated_tokens)} tokens). ")
            return True
            
        return False

    def generate_next_output(self, temperature, max_new_tokens, top_p, top_k, min_p):
        backend = self.agent._get_backend_for_level(self.call_agent_level)
        last_outputs_str, self.token_overflow = self.agent._generate_with_retry(
            conversation=self.conversation,
            tools=self.picked_tools_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p, top_k=top_k, min_p=min_p,
            backend=backend,
            # Per-request cancel signal (set on ctx by run_multistep_agent);
            # enables streaming generation that aborts mid-token on client
            # disconnect instead of running to full max_new_tokens.
            cancel_event=getattr(self, "cancel_event", None))
        return last_outputs_str

    def _handle_token_overflow(self, temperature, max_new_tokens, top_p, top_k, min_p, return_conversation):
        """Reactive Context Compaction (Strategy 2) for when generation still fails."""
        logger.info("Internal error: The number of tokens exceeds the maximum limit despite tool shrink. Attempting context checkpoint...")
        backend = self.agent._get_backend_for_level(self.call_agent_level)
        
        context_summary = self.agent.run_context_summary_agent(
            self.conversation,
            temperature=0.1,
            max_new_tokens=20480,
            top_p=top_p, top_k=top_k, min_p=min_p
        )

        if context_summary:
            logger.info("Context checkpoint successful. Updating conversation history.")
            new_conversation = []
            if self.conversation and self.conversation[0].get('role') == 'system':
                new_conversation.append(self.conversation[0])
            else:
                if backend != 'athena':
                    new_conversation = self.agent.set_system_prompt(new_conversation, self.agent.gpt_planning_prompt)
                else:
                    new_conversation = self.agent.set_system_prompt(new_conversation, self.agent.prompt_multi_step)

            first_user_msg = next((msg for msg in self.conversation if msg.get('role') == 'user'), None)
            if first_user_msg:
                new_conversation.append(first_user_msg)

            summary_message = f"Here is a summary of the progress so far (Context Checkpoint):\n\n{context_summary}\n\nPlease continue the task from here."
            new_conversation.append({"role": "user", "content": summary_message})
            self.conversation[:] = new_conversation
            return True

        logger.info(" Failed to generate context summary.")
        return False

    def check_final_answer(self, last_outputs_str, return_conversation):
        if '[FinalAnswer]' in last_outputs_str:
            final_answer = last_outputs_str.split('[FinalAnswer]')[-1].strip()
            self.conversation.append({"role": "assistant", "content": final_answer})
            return True, (final_answer, self.conversation) if return_conversation else final_answer

        # If model outputs </think> without <tool_call>, treat as final answer
        if '</think>' in last_outputs_str and '<tool_call>' not in last_outputs_str:
            if not hasattr(self, '_no_tool_rounds'):
                self._no_tool_rounds = 0
            self._no_tool_rounds += 1
            if self._no_tool_rounds >= 2:
                logger.info(f"[anti-repetition] {self._no_tool_rounds} consecutive rounds without tool_call. Treating as final answer.")
                final_answer = last_outputs_str.split('</think>')[-1].strip()
                if not final_answer:
                    final_answer = last_outputs_str
                self.conversation.append({"role": "assistant", "content": final_answer})
                return True, (final_answer, self.conversation) if return_conversation else final_answer
        else:
            self._no_tool_rounds = 0

        return False, None

    def handle_force_finish(self, temperature, max_new_tokens, top_p, top_k, min_p, return_conversation):
        if self.agent.force_finish:
            answer = self.agent.get_answer_based_on_unfinished_reasoning(
                self.conversation, temperature, max_new_tokens,
                top_p=top_p, top_k=top_k, min_p=min_p)
            return (answer, self.conversation) if return_conversation else answer
        return (None, self.conversation) if return_conversation else None

    def _emergency_final_answer(self, temperature, max_new_tokens, top_p, top_k, min_p):
        """Last-resort answer synthesis that NEVER yields a bare error.

        Reached only when Strategy 1 + Strategy 2 context management have both
        failed to fit the prompt (typically a tool returned a huge output).
        Tiers down until SOMETHING answers the user:
          1. Hard-truncate every tool/function message, then synthesize a final
             answer from the trimmed conversation.
          2. Drop ALL tool messages — synthesize from system + the question.
          3. Direct no-tools answer from the question alone.
          4. Graceful honest message (still not a raw stack-trace-style error).
        Always returns a non-empty string. Uses the blocking inference path
        (no cancel_event) so it completes even mid-overflow.
        """
        agent = self.agent

        def _synth(conv):
            try:
                ans = agent.get_answer_based_on_unfinished_reasoning(
                    conv, temperature, min(max_new_tokens, 1024),
                    top_p=top_p, top_k=top_k, min_p=min_p)
                ans = (ans or "").strip()
                if "[FinalAnswer]" in ans:
                    ans = ans.split("[FinalAnswer]")[-1].strip()
                if "</think>" in ans:
                    ans = ans.split("</think>")[-1].strip()
                return ans
            except Exception as e:  # noqa: BLE001
                logger.info(f"[emergency] synthesize failed: {e!r}")
                return ""

        # Tier 1 — hard-truncate oversized tool outputs (the usual overflow
        # cause) and synthesize from what remains.
        CAP = 1200
        trimmed = []
        for m in self.conversation:
            if isinstance(m, dict) and m.get("role") in ("tool", "function"):
                c = str(m.get("content", ""))
                if len(c) > CAP:
                    m = {**m, "content": c[:CAP] + " …[truncated for length]"}
            trimmed.append(m)
        self.conversation = trimmed
        ans = _synth(self.conversation)
        if ans:
            return ans

        # Tier 2 — drop all tool messages; keep system prompt + the question.
        minimal = [m for m in self.conversation
                   if isinstance(m, dict) and m.get("role") == "system"]
        minimal.append({"role": "user", "content": str(self.message)})
        ans = _synth(minimal)
        if ans:
            return ans

        # Tier 3 — direct, no-tools answer from the question alone.
        try:
            direct = agent.llm_infer(
                messages=[{
                    "role": "user",
                    "content": ("Answer the following clinical question "
                                "concisely from general medical knowledge:\n\n"
                                + str(self.message)),
                }],
                temperature=temperature, tools=None,
                max_new_tokens=min(max_new_tokens, 768),
                top_p=top_p, top_k=top_k, min_p=min_p)
            direct = (direct or "").strip()
            if "</think>" in direct:
                direct = direct.split("</think>")[-1].strip()
            if direct:
                return direct
        except Exception as e:  # noqa: BLE001
            logger.info(f"[emergency] direct answer failed: {e!r}")

        # Tier 4 — graceful honest fallback (never a raw internal error).
        return ("I couldn't complete full tool-based verification for this "
                "question because the gathered information exceeded the model's "
                "context limit. Based on the reasoning so far I don't have a "
                "confident verified answer — please try a more specific "
                "question or narrow the scope.")


class AthenaCore:
    """Internal multi-step reasoning engine. Use the ``AthenaR1`` wrapper instead.

    This is a module-private implementation detail; external users should always
    construct an ``athena_r1.AthenaR1`` instance, which wraps this engine in a
    clean, supported interface.
    """

    def __init__(self, model_name,
                 rag_model_name,
                 tool_files_dict=None,
                 enable_finish=True,
                 enable_rag=True,
                 enable_summary=False,
                 init_rag_num=0,
                 step_rag_num=10,
                 force_finish=True,
                 additional_default_tools=None,
                 cache_tool=False,
                 tool_in_results=False,
                 add_call_id=True,
                 vllm_server_url='http://127.0.0.1:8000',
                 max_token=32768,
                 compress_ratio=3.5,
                 tool_call_max_retries=3,
                 keep_full_history=False,
                 backend_by_level: Optional[Dict[int, str]] = None,
                 backend_default: str = "athena",
                 backend_models: Optional[Dict[str, str]] = None,
                 max_agent_level: Optional[int] = None,
                 summary_format_vllm_server_url=None,
                 summary_format_model_name=None,
                 summary_format_args=None,
                 update_tool_list=False,
                 ):
        # Honor the constructor argument. An earlier version hardcoded 0 here
        # and relied on the AthenaR1 wrapper to patch `_core.max_agent_level`
        # right after construction — but constructing the engine directly
        # (some eval scripts do) would then silently lose the depth setting.
        self.max_agent_level = 0 if max_agent_level is None else int(max_agent_level)
        self.update_tool_list = update_tool_list
        self.model_name = model_name
        self.tokenizer = None
        self.tool_files_dict = tool_files_dict
        self.model = None
        self.tooluniverse = None

        # System prompts live at module scope (see _prompts.py) to keep this
        # constructor readable.
        self.prompt_multi_step = PROMPT_MULTI_STEP
        self.gpt_planning_prompt = GPT_PLANNING_PROMPT
        self.context_summary_prompt = CONTEXT_SUMMARY_PROMPT

        self.enable_finish = enable_finish
        self.enable_rag = enable_rag
        self.enable_summary = enable_summary
        self.init_rag_num = init_rag_num
        self.step_rag_num = step_rag_num
        self.force_finish = force_finish
        self.additional_default_tools = additional_default_tools
        self.cache_tool = cache_tool
        self.tool_in_results = tool_in_results
        self.add_call_id = add_call_id
        # Format / mode are fixed for ATHENA-R1: Qwen3-8B served via vLLM HTTP.
        self.format = 'qwen'
        self.vllm_server_url = vllm_server_url
        self.max_token = max_token
        self.compress_ratio = compress_ratio
        self.tool_call_max_retries = tool_call_max_retries
        self.keep_full_history = keep_full_history

        # Backend selection config
        self.backend_default = backend_default
        self.backend_by_level = dict(backend_by_level or {})
        # Model id per external (non-local) backend. Callers override individual
        # entries via `backend_models`; the rest keep these defaults.
        self.backend_models = {
            "gpt": "gpt-5",
            "claude": "claude-opus-4-8",
            "gemini": "gemini-2.5-pro",
        }
        self.backend_models.update(backend_models or {})
        self.summary_format_vllm_server_url = summary_format_vllm_server_url
        self.summary_format_model_name = summary_format_model_name
        self.summary_format_args = summary_format_args or {
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': 20,
            'min_p': 0.0,
            'max_new_tokens': 1024  # was 32768 — too large; native_v2 only needs short letter answer
        }
        self.summary_format_model = None
        self.tool_processor_registry = None

        # Azure OpenAI client is constructed lazily on first use. Users who
        # only invoke `Backend.ATHENA` paths don't need Azure credentials —
        # crashing the whole constructor on a missing key (as `AzureOpenAI()`
        # does) would be hostile. We retain the env-var resolution here but
        # defer the actual client object to a property.
        self._gpt_client_obj: Optional[AzureOpenAI] = None
        # Anthropic (Claude) + Gemini clients, likewise lazy.
        self._claude_client_obj = None
        self._gemini_client_obj: Optional[OpenAI] = None

    @property
    def gpt_client(self) -> AzureOpenAI:
        """Lazy-initialized Azure OpenAI client.

        Accepts both ``AZURE_API_KEY`` (public wrapper convention) and
        ``AZURE_OPENAI_API_KEY`` (Azure SDK convention). Raises a clear
        error only on first GPT use if no credentials are configured.
        """
        if self._gpt_client_obj is None:
            api_key = (
                os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
            )
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not api_key:
                raise RuntimeError(
                    "Azure OpenAI API key is required for Backend.GPT or "
                    "map_to_option(backend=Backend.GPT). Set the "
                    "`AZURE_API_KEY` environment variable, pass "
                    "`azure_api_key=...` to AthenaR1(), or stick to "
                    "Backend.ATHENA which uses your local vLLM-served model."
                )
            if not azure_endpoint:
                raise RuntimeError(
                    "Azure OpenAI endpoint is required for Backend.GPT. Set "
                    "the `AZURE_OPENAI_ENDPOINT` environment variable or pass "
                    "`azure_endpoint=...` to AthenaR1()."
                )
            self._gpt_client_obj = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=os.getenv(
                    "AZURE_OPENAI_API_VERSION", "2025-04-01-preview"
                ),
            )
        return self._gpt_client_obj

    @property
    def claude_client(self):
        """Lazy Anthropic client for ``Backend.CLAUDE``.

        The Anthropic SDK resolves credentials from ``ANTHROPIC_API_KEY`` or an
        ``ant auth`` profile, so we don't pre-check a key here — a clear error
        surfaces on first use if none is configured. ``anthropic`` is imported
        lazily so the other backends don't require it installed.
        """
        if self._claude_client_obj is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "Backend.CLAUDE needs the `anthropic` package. Install it "
                    'with `pip install "athena-r1[api]"` or `pip install anthropic`.'
                ) from exc
            self._claude_client_obj = anthropic.Anthropic()
        return self._claude_client_obj

    @property
    def gemini_client(self) -> OpenAI:
        """Lazy client for ``Backend.GEMINI`` via Google's OpenAI-compatible API.

        Uses the same ``openai`` client ATHENA already depends on, pointed at
        Google's endpoint. Override the base URL with ``GEMINI_BASE_URL``.
        """
        if self._gemini_client_obj is None:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "Backend.GEMINI requires a `GEMINI_API_KEY` environment "
                    "variable (Google AI Studio key)."
                )
            base_url = os.getenv(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._gemini_client_obj = OpenAI(api_key=api_key, base_url=base_url)
        return self._gemini_client_obj

    def init_model(self, tool_type: List[str]) -> None:
        """Load ToolUniverse + the vLLM client. Idempotent — call once."""
        self.load_tooluniverse(tool_type)
        self.load_models()

    def load_models(self, model_name=None):
        if model_name is not None:
            if model_name == self.model_name:
                return f"The model {model_name} is already loaded."
            self.model_name = model_name

        if self.max_token is None:
            self.max_token = 40960

        # External API backends (Claude/Gemini/GPT) serve no local weights and
        # pass model_name=None. Skip the vLLM client + tokenizer load entirely —
        # those are only used by the local (Backend.ATHENA) path.
        if self.model_name is None:
            logger.info(
                "No local model_name set — skipping vLLM/tokenizer load "
                "(external API backend)."
            )
            self.model = None
            self.tokenizer = None
            self.chat_template = None
            return

        if not self.vllm_server_url.endswith('/v1'):
            self.vllm_server_url = f"{self.vllm_server_url.rstrip('/')}/v1"
        # Per-request timeout for the vLLM client. The OpenAI SDK defaults to a
        # 600 s read timeout, so an occasional stuck vLLM request (one that never
        # returns its first token under concurrency) would block a whole question
        # for ~10 min before failing. A tighter read timeout makes such a hang
        # surface in seconds so the round loop / caller can retry instead of
        # losing the question. Override with ATHENA_VLLM_READ_TIMEOUT (seconds;
        # set 0 to restore the SDK default).
        _read_to = float(os.environ.get("ATHENA_VLLM_READ_TIMEOUT", "120"))
        _client_timeout = (
            httpx.Timeout(connect=10.0, read=_read_to, write=_read_to, pool=_read_to)
            if _read_to > 0 else None
        )
        self.model = OpenAI(api_key="EMPTY", base_url=self.vllm_server_url,
                            timeout=_client_timeout)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.chat_template = Template(self.tokenizer.chat_template)

        if self.summary_format_vllm_server_url is not None:
            if not self.summary_format_vllm_server_url.endswith('/v1'):
                self.summary_format_vllm_server_url = f"{self.summary_format_vllm_server_url.rstrip('/')}/v1"
            self.summary_format_model = OpenAI(
                api_key="EMPTY",
                base_url=self.summary_format_vllm_server_url,
            )
            if self.summary_format_model_name is None:
                self.summary_format_model_name = self.model_name
            logger.info(f"Summary/Format model loaded from {self.summary_format_vllm_server_url} with model {self.summary_format_model_name}")

        return f"Model {model_name} loaded successfully."

    def load_tooluniverse(self, tool_type):
        # If TOOLUNIVERSE_API env set, use HTTP client (shared server, low memory).
        # Otherwise fall back to local ToolUniverse (high memory per instance).
        api_url = os.environ.get("TOOLUNIVERSE_API")
        if api_url:
            from athena_r1._tu_compat import CompatToolUniverseClient
            if not api_url.startswith("http"):
                api_url = "http://" + api_url
            logger.info(f"Using ToolUniverse server: {api_url}")
            self.tooluniverse = CompatToolUniverseClient(api_url)
        else:
            self.tooluniverse = ToolUniverse(tool_files=self.tool_files_dict)
        if tool_type is not None:
            if 'tool_finder' not in tool_type:
                tool_type.append("tool_finder")
            self.tooluniverse.load_tools(tool_type=tool_type)
        else:
            logger.info("load all available tools in tooluniverse")
            self.tooluniverse.load_tools()

        # Works for both local ToolUniverse (attribute) and ToolUniverseClient (HTTP):
        self.tool_processor_registry = ToolProcessorRegistry(self)
        logger.info("ToolUniverse loaded successfully")

    def initialize_tools_prompt(self, call_agent, call_agent_level, message):
        picked_tools_prompt = []
        if self.max_agent_level is not None:
            call_agent = bool(call_agent) and (call_agent_level < self.max_agent_level)
        picked_tools_prompt = self.add_special_tools(picked_tools_prompt, call_agent=call_agent, call_agent_level=call_agent_level)

        if not call_agent and self.enable_rag:
            # Call Tool_RAG
            tool_call = {
                "name": "Tool_RAG",
                "arguments": {
                    "description": message,
                    "limit": self.init_rag_num
                }
            }
            try:
                call_result = self.tooluniverse.run_one_function(tool_call)
                if isinstance(call_result, dict) and 'error' in call_result:
                    logger.info(f"Tool_RAG Error: {call_result}")
                    return picked_tools_prompt, call_agent_level

                # Parse the result
                if isinstance(call_result, str):
                    try:
                        new_tools = json.loads(call_result)
                    except (json.JSONDecodeError, TypeError):
                        new_tools = []
                elif isinstance(call_result, list):
                    new_tools = call_result
                else:
                    new_tools = []

                if new_tools:
                    new_tools_prompt = self.tooluniverse.prepare_tool_prompts(new_tools)
                    picked_tools_prompt += new_tools_prompt
            except Exception as e:
                logger.info(f"Tool_RAG Exception: {e}")
                traceback.print_exc()

        return picked_tools_prompt, call_agent_level

    def add_special_tools(self, tools, call_agent=False, call_agent_level=0):
        # The CallAgent tool is exposed to the model only when call_agent=True
        # AND the configured depth limit hasn't been reached. Otherwise the
        # model never sees it and can't recurse.
        if call_agent and (
            self.max_agent_level is None or call_agent_level < self.max_agent_level
        ):
            tools.append(self.tooluniverse.tool_specification('CallAgent', return_prompt=True))
            logger.info("CallAgent tool is added")
        if not call_agent:
            if self.enable_rag:
                tools.append(self.tooluniverse.tool_specification('Tool_RAG', return_prompt=True))
                logger.info("Tool_RAG tool is added")

            if self.additional_default_tools is not None:
                for each_tool_name in self.additional_default_tools:
                    tool_prompt = self.tooluniverse.tool_specification(each_tool_name, return_prompt=True)
                    if tool_prompt is not None:
                        logger.info(f"{each_tool_name} tool is added")
                        tools.append(tool_prompt)
        if self.enable_finish:
            tools.append(self.tooluniverse.tool_specification('Finish', return_prompt=True))
            logger.info("Finish tool is added")
        return tools

    def set_system_prompt(self, conversation, sys_prompt):
        if len(conversation) == 0:
            conversation.append({"role": "system", "content": sys_prompt})
        else:
            conversation[0] = {"role": "system", "content": sys_prompt}
        return conversation

    def _clean_conversation_keep_final_answer(self, conversation: list) -> list:
        """Clean conversation by removing reasoning traces, keeping only final answers."""
        if not conversation or len(conversation) < 2:
            return conversation

        # Find the last user message index
        last_user_idx = None
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return conversation

        cleaned = conversation[:last_user_idx + 1]

        for i in range(last_user_idx + 1, len(conversation)):
            msg = conversation[i]
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if "[FinalAnswer]" in content:
                    final_answer = content.split("[FinalAnswer]")[-1].strip()
                    cleaned.append({"role": "assistant", "content": final_answer})
                elif msg.get("role") == "assistant" and i == len(conversation) - 1:
                    cleaned.append(msg)
            elif msg.get("role") != "tool":
                cleaned.append(msg)

        return cleaned

    def initialize_conversation(self, message, conversation=None, call_agent_level: int = 0):
        """Initialize or extend a clean conversation for LLM."""
        if conversation is not None and len(conversation) > 0:
            if not self.keep_full_history:
                conversation = self._clean_conversation_keep_final_answer(conversation)
            conversation = list(conversation)
        else:
            conversation = []
            backend = self._get_backend_for_level(call_agent_level)
            if backend != "athena":
                sys_prompt = self.gpt_planning_prompt
            else:
                sys_prompt = self.prompt_multi_step
            conversation = self.set_system_prompt(conversation, sys_prompt)

        # Append current user message if not already present
        if not conversation or conversation[-1].get("role") != "user" or conversation[-1].get("content") != message:
            conversation.append({"role": "user", "content": message})

        return conversation

    def run_function_call(self, fcall_str,
                         existing_tools_prompt=None,
                         message_for_call_agent=None,
                         call_agent=False,
                         call_agent_level=None,
                         temperature=None,
                         conversation=None,
                         enable_summary=False,
                         last_status=None,
                         max_new_tokens=4096,
                         top_p=0.95,
                         top_k=20,
                         min_p=0.0,
                         progress_callback=None,
                         agent_id="main",
                         cancel_event=None):
        """Dispatch each <tool_call> to its registered processor.

        `progress_callback`, `agent_id`, and `cancel_event` are forwarded via
        the per-call context so multi-agent / Tool_RAG processors can emit
        nested streaming events AND propagate cancellation to child agents.
        """

        function_call_json, message = self.tooluniverse.extract_function_call_json(
            fcall_str, return_message=True, verbose=False, format=self.format)

        call_results = []
        special_tool_call = ''
        callagent_plans = []

        if function_call_json is not None:
            if isinstance(function_call_json, list):
                if not self.cache_tool and self.update_tool_list:
                    if 'Tool_RAG' in [each['name'] for each in function_call_json]:
                        existing_tools_prompt = self.add_special_tools([], call_agent=call_agent, call_agent_level=call_agent_level or 0)

                # Build context for tool processors
                context = {
                    'message': message,
                    'existing_tools_prompt': existing_tools_prompt,
                    'conversation': conversation,
                    'call_agent': call_agent,
                    'call_agent_level': call_agent_level,
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': top_k,
                    'min_p': min_p,
                    'callagent_plans': callagent_plans,
                    'update_tool_list': self.update_tool_list,
                    # For multi-agent + Tool_RAG nested streaming:
                    'progress_callback': progress_callback,
                    'agent_id': agent_id,
                    'max_new_tokens': max_new_tokens,
                    # Threaded through so CallAgent can propagate user-abort
                    # to recursive child runs (otherwise sub-agents keep
                    # running after the top-level Stop button is clicked).
                    'cancel_event': cancel_event,
                }

                # Pre-allocate result slots so parallel CallAgents can write
                # into their original positions (preserving call/result order).
                slotted_results: list[dict | None] = [None] * len(function_call_json)

                def _dispatch(idx: int):
                    """Run one processor; return (idx, call_result, context_updates) or None if skipped."""
                    fc = function_call_json[idx]
                    if not isinstance(fc, dict) or "name" not in fc:
                        logger.info(f"[malformed tool_call skipped] {fc}")
                        return idx, None, None
                    proc = self.tool_processor_registry.get_processor(fc["name"])
                    result, updates = proc.process(fc, context)
                    return idx, result, updates

                def _commit(idx: int, call_result, context_updates) -> bool:
                    """Apply context updates + build the tool-result message. Returns True if Finish."""
                    nonlocal special_tool_call, callagent_plans, existing_tools_prompt
                    if context_updates is None:
                        return False
                    context.update(context_updates)
                    callagent_plans = context.get('callagent_plans', callagent_plans)
                    existing_tools_prompt = context.get('existing_tools_prompt', existing_tools_prompt)
                    special_tool_call = context.get('special_tool_call', special_tool_call)
                    if call_result is None:
                        logger.info(f"This call returns None: {function_call_json[idx]}")
                    call_id = None
                    if self.add_call_id:
                        call_id = self.tooluniverse.call_id_gen()
                        function_call_json[idx]["call_id"] = call_id
                    slotted_results[idx] = self._build_tool_result_message(call_result, call_id)
                    return special_tool_call == 'Finish'

                # Parallelization policy: consecutive CallAgent tool-calls are
                # independent (each spawns its own subagent recursion) and can
                # run concurrently. Other tool kinds run sequentially because
                # they mutate shared state (`existing_tools_prompt`) or alter
                # control flow (`Finish`).
                # Disable parallelism with ATHENA_R1_PARALLEL_CALLAGENT=0.
                import os as _os
                parallel_enabled = _os.environ.get("ATHENA_R1_PARALLEL_CALLAGENT", "1") not in ("0", "false", "False")

                i = 0
                stop_loop = False
                while i < len(function_call_json) and not stop_loop:
                    fc = function_call_json[i]
                    is_callagent = isinstance(fc, dict) and fc.get("name") == "CallAgent"
                    if not (parallel_enabled and is_callagent):
                        # Sequential path.
                        idx, result, updates = _dispatch(i)
                        if _commit(idx, result, updates):
                            stop_loop = True
                        i += 1
                        continue
                    # Gather the run of consecutive CallAgents starting at i.
                    j = i
                    while (
                        j < len(function_call_json)
                        and isinstance(function_call_json[j], dict)
                        and function_call_json[j].get("name") == "CallAgent"
                    ):
                        j += 1
                    batch = list(range(i, j))
                    if len(batch) == 1:
                        idx, result, updates = _dispatch(batch[0])
                        if _commit(idx, result, updates):
                            stop_loop = True
                    else:
                        from concurrent.futures import ThreadPoolExecutor, as_completed
                        # Cap concurrency to avoid runaway thread explosion when
                        # a model emits dozens of CallAgents in one batch (each
                        # of which may spawn more recursively). 4 parallel is
                        # plenty for typical interleaved sub-agent flows; the
                        # rest queue up behind them.
                        _MAX_PARALLEL_CALLAGENT = int(_os.environ.get(
                            "ATHENA_R1_PARALLEL_CALLAGENT_MAX", "4"))
                        workers = min(len(batch), max(1, _MAX_PARALLEL_CALLAGENT))
                        logger.info(f"Parallelizing {len(batch)} CallAgent calls (max {workers} concurrent)")
                        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="callagent") as ex:
                            futures = {ex.submit(_dispatch, idx): idx for idx in batch}
                            # Collect completions; commit in original index order
                            # so tool-result messages line up with their calls.
                            completed: dict[int, tuple] = {}
                            for fut in as_completed(futures):
                                idx, result, updates = fut.result()
                                completed[idx] = (result, updates)
                            for idx in batch:
                                result, updates = completed[idx]
                                if _commit(idx, result, updates):
                                    stop_loop = True
                                    break
                    i = j

                # Compact slotted_results (drop None entries from skipped/malformed
                # calls and from positions we never reached after Finish).
                call_results = [r for r in slotted_results if r is not None]
        else:
            call_results.append({
                "role": "tool",
                "content": json.dumps({"content": "Not a valid function call, please check the function call format."})
            })

        revised_messages = [{
            "role": "assistant",
            "content": message.strip(),
            "tool_calls": json.dumps(function_call_json)
        }] + call_results



        should_stop = False
        final_result = None

        if special_tool_call == 'Finish':
            should_stop = True
            final_result = revised_messages[0]['content']
        elif special_tool_call in ['RequireClarification', 'DirectResponse']:
            should_stop = True
            final_result = call_results[0]['content'] if call_results else ''

        return {
            'messages': revised_messages,
            'tools_prompt': existing_tools_prompt,
            'special_tool_call': special_tool_call,
            'should_stop': should_stop,
            'final_result': final_result
        }

    def _build_tool_result_message(self, call_result: str, call_id: str = None) -> dict:
        """Build a tool result message dict with optional call_id."""
        if call_id:
            return {
                "role": "tool",
                "content": json.dumps({"content": call_result, "call_id": call_id})
            }
        else:
            return {
                "role": "tool",
                "content": json.dumps({"content": call_result})
            }

    def llm_infer(self, messages, temperature=0.1, tools=None,
                  output_begin_string=None, max_new_tokens=2048,
                  model=None, seed=None,
                  check_token_status=False, top_p=0.95, top_k=20, min_p=0.0,
                  model_name=None, repetition_penalty=None,
                  presence_penalty=1.5, cancel_event=None):
        # Qwen3-8B HF recommendation: top_p=0.95, top_k=20, min_p=0, T=0.6 (thinking),
        # presence_penalty 0-2 to suppress endless repetition. No repetition_penalty.
        # Env-var override for ablation runs:
        if os.environ.get('INFER_PRESENCE_PENALTY') is not None:
            v = float(os.environ['INFER_PRESENCE_PENALTY'])
            presence_penalty = v if v > 0 else None
        if os.environ.get('INFER_REPETITION_PENALTY') is not None:
            v = float(os.environ['INFER_REPETITION_PENALTY'])
            repetition_penalty = v if v > 0 else None

        def filter_message_keys(messages):
            allowed_keys = {"role", "content", "tools", "tool_calls"}
            filtered = []
            for msg in messages:
                filtered.append({k: v for k, v in msg.items() if k in allowed_keys and v is not None})
            return filtered

        messages = copy.deepcopy(messages)
        messages = filter_message_keys(messages)
        messages = convert_to_qwen_format(messages)

        prompt = self.chat_template.render(
            messages=messages, tools=tools, add_generation_prompt=True)
        if output_begin_string is not None:
            prompt += output_begin_string

        if check_token_status and self.max_token is not None:
            token_overflow = False
            num_input_tokens = len(self.tokenizer.encode(prompt, return_tensors="pt")[0])
            if self.max_token is not None:
                if num_input_tokens > self.max_token:
                    gc.collect()
                    logger.info(f"Number of input tokens before inference: {num_input_tokens}")
                    token_overflow = True
                    return None, token_overflow

        is_summary_call = output_begin_string and "</think>" in output_begin_string

        if model is None:
            model = self.model
        if model_name is None:
            model_name = self.model_name

        extra_body = {"top_k": top_k, "min_p": min_p}
        if repetition_penalty is not None:
            extra_body["repetition_penalty"] = repetition_penalty

        kwargs = {
            "model": model_name,
            "prompt": prompt,
            "top_p": top_p,
            "temperature": temperature,
            "extra_body": extra_body,
        }
        # Qwen3 HF recommendation: presence_penalty 0-2 to reduce endless repetition.
        if presence_penalty is not None and presence_penalty != 0:
            kwargs["presence_penalty"] = presence_penalty
        if max_new_tokens is not None:
            kwargs["max_tokens"] = max_new_tokens
        if is_summary_call:
            kwargs["stop"] = ["</think>"]

        # Retry transient vLLM failures (timeouts / connection drops). With the
        # client read-timeout above, an occasionally-stuck request now fails in
        # ~2 min instead of ~10; retrying it almost always succeeds, so a single
        # bad request no longer loses the whole question. Honors cancel_event
        # between attempts so a disconnected client still aborts fast.
        _max_attempts = int(os.environ.get("ATHENA_VLLM_MAX_RETRIES", "3"))
        output = ""
        _api_failed = False
        for _attempt in range(_max_attempts):
            if cancel_event is not None and cancel_event.is_set():
                break
            try:
                if cancel_event is not None:
                    # Streaming generation so the engine can abort mid-token when
                    # the client disconnects. The blocking (non-stream) path runs
                    # to full max_new_tokens before the round loop can re-check
                    # cancellation — for a verbose answer that's 40-50s of wasted
                    # GPU per in-flight round. Here we poll cancel_event every
                    # chunk and, on cancel, close the stream: dropping the HTTP
                    # connection makes vLLM abort the sequence server-side and
                    # free the GPU immediately. In the non-cancelled case the
                    # concatenated chunks are byte-identical to the blocking
                    # result, so output is unchanged for normal completions.
                    kwargs["stream"] = True
                    parts = []
                    stream = model.completions.create(**kwargs)
                    # First-chunk watchdog. The SDK blocks inside the `for`
                    # below on the first __next__() until vLLM emits chunk 1.
                    # httpx's read-timeout guards inter-chunk gaps but has been
                    # seen NOT to fire when vLLM accepts the request yet never
                    # schedules the sequence (no first chunk, no error, socket
                    # held open) — that hangs the whole question. A one-shot
                    # timer closes the stream if chunk 1 never arrives; the
                    # close drops the connection so __next__() raises and we
                    # fall through to the retry below. Once generation is
                    # underway we retire the watchdog and let the read-timeout
                    # cover the (working) inter-chunk case.
                    _first_chunk_to = float(os.environ.get(
                        "ATHENA_VLLM_FIRST_CHUNK_TIMEOUT", "90"))
                    _wd_lock = threading.Lock()
                    _wd_closed = [False]

                    # Bind this iteration's stream/lock/flag as defaults so the
                    # timer never touches a later attempt's reassigned objects.
                    def _close_stuck_stream(_s=stream, _lock=_wd_lock, _closed=_wd_closed):
                        with _lock:
                            if not _closed[0]:
                                _closed[0] = True
                                try:
                                    _s.close()
                                except Exception:  # noqa: BLE001
                                    pass
                    _timer = (threading.Timer(_first_chunk_to, _close_stuck_stream)
                              if _first_chunk_to > 0 else None)
                    if _timer is not None:
                        _timer.daemon = True
                        _timer.start()
                    try:
                        _first = True
                        for ch in stream:
                            if _first:
                                if _timer is not None:
                                    _timer.cancel()
                                _first = False
                            if cancel_event.is_set():
                                break
                            if getattr(ch, "choices", None):
                                piece = ch.choices[0].text
                                if piece:
                                    parts.append(piece)
                    finally:
                        if _timer is not None:
                            _timer.cancel()
                        try:
                            stream.close()
                        except Exception:  # noqa: BLE001
                            pass
                    output = "".join(parts).strip()
                else:
                    output_raw = model.completions.create(**kwargs)
                    output = output_raw.choices[0].text.strip()
                _api_failed = False
                break
            except Exception as e:  # noqa: BLE001
                _api_failed = True
                output = ""
                logger.info(
                    f"[llm_infer] API error (attempt {_attempt + 1}/{_max_attempts}): {e}")
        if _api_failed and check_token_status:
            return None, True

        logger.info("" + output + "")
        if check_token_status and self.max_token is not None:
            return output, token_overflow
        return output

    def _get_backend_for_level(self, level: int) -> str:
        """Return backend name for a given agent nesting level."""
        backend = self.backend_by_level.get(level, self.backend_default) if hasattr(self, "backend_by_level") else self.backend_default
        if backend is None:
            backend = "athena"
        backend = str(backend).strip().lower()
        if backend in ("original", "vllm", "local"):
            backend = "athena"
        return backend

    def gpt_llm_infer(self, messages: List[Dict[str, Any]],
                      temperature: float = 1.0,
                      tools: Any = None,
                      max_new_tokens: int = 8192,
                      top_p: float = 1.0,
                      backend: str = "gpt") -> str:
        """OpenAI-compatible chat backend inference (GPT via Azure, or Gemini).

        Tools are serialized into the system prompt (the ATHENA-R1 prompt
        protocol), and any native ``tool_calls`` the model returns are converted
        to ``<tool_call>`` strings so the round loop parses them uniformly.
        """
        # Convert conversation for GPT
        gpt_messages = self._convert_conversation_for_gpt(messages)
        gpt_messages_wo_system = [m for m in gpt_messages if m.get("role") != "system"]

        tools_blob = self._serialize_tools_for_gpt(tools)
        system_prompt = (
            f"{self.gpt_planning_prompt}\n\n"
            "## Available tools (serialized)\n"
            f"{tools_blob}\n"
        )

        request_messages = [{"role": "system", "content": system_prompt}] + gpt_messages_wo_system

        client = self.gemini_client if backend == "gemini" else self.gpt_client
        model = self.backend_models.get(backend, "gpt-5")
        responses = client.chat.completions.create(
            model=model,
            messages=request_messages,
        )
        msg = responses.choices[0].message
        raw_content = msg.content
        answer = (raw_content or "").strip()
        logger.info(f"{backend}: {answer}")

        # Convert OpenAI tool_calls schema to the ATHENA-R1 <tool_call> string format.
        try:
            tool_calls = getattr(msg, "tool_calls", None)
        except Exception:
            tool_calls = None

        if tool_calls and "<tool_call>" not in answer:
            blocks: List[str] = []
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn is not None else None
                args_raw = getattr(fn, "arguments", None) if fn is not None else None
                if not name:
                    continue
                args_obj: Any = {}
                if isinstance(args_raw, str) and args_raw.strip():
                    try:
                        args_obj = json.loads(args_raw)
                    except Exception:
                        args_obj = {"_raw": args_raw}
                # JSON-encode `name` so embedded quotes/backslashes can't
                # break out of the string and corrupt the surrounding tool_call
                # JSON. `json.dumps` adds the wrapping quotes, so this slot
                # is no longer enclosed in literal "..." in the f-string.
                blocks.append(
                    f'<tool_call>{{"name": {json.dumps(name, ensure_ascii=False)}, '
                    f'"arguments": {json.dumps(args_obj, ensure_ascii=False)}}}</tool_call>'
                )

            if blocks:
                prefix = answer.strip()
                if prefix:
                    return f"{prefix}\n" + "\n".join(blocks)
                return "\n".join(blocks)

        return answer

    def claude_llm_infer(self, messages: List[Dict[str, Any]],
                         temperature: float = 1.0,
                         tools: Any = None,
                         max_new_tokens: int = 8192,
                         top_p: float = 1.0) -> str:
        """Claude backend inference via the official Anthropic SDK.

        Mirrors :meth:`gpt_llm_infer`: the tool set is serialized into the
        system prompt (the ATHENA-R1 prompt protocol) and Claude — a strong
        instruction follower — emits ``<tool_call>`` blocks in its text, which
        the round loop parses uniformly. No native Anthropic tool schema is
        passed, keeping the loop backend-agnostic.
        """
        conv = self._convert_conversation_for_gpt(messages)
        chat_msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in conv
            if m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()
        ]
        # Anthropic requires the first message to be a user turn.
        if not chat_msgs or chat_msgs[0]["role"] != "user":
            chat_msgs.insert(0, {"role": "user", "content": "Begin."})

        tools_blob = self._serialize_tools_for_gpt(tools)
        system_prompt = (
            f"{self.gpt_planning_prompt}\n\n"
            "## Available tools (serialized)\n"
            f"{tools_blob}\n"
        )

        model = self.backend_models.get("claude", "claude-opus-4-8")
        resp = self.claude_client.messages.create(
            model=model,
            max_tokens=max(1024, int(max_new_tokens)),
            system=system_prompt,
            messages=chat_msgs,
        )
        # Guard the refusal stop reason before reading content blocks.
        if getattr(resp, "stop_reason", None) == "refusal":
            logger.info("Claude declined the request (stop_reason=refusal).")
            return "The model declined to answer this request."
        answer = "".join(
            getattr(b, "text", "") for b in resp.content
            if getattr(b, "type", None) == "text"
        ).strip()
        logger.info(f"claude: {answer}")
        return answer

    def _serialize_tools_for_gpt(self, tools: Any) -> str:
        """Serialize tool prompt objects into a string for GPT context."""
        if tools is None:
            return "[]"
        try:
            return json.dumps(tools, ensure_ascii=False, indent=1, default=str)
        except Exception:
            return str(tools)

    def _convert_conversation_for_gpt(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert internal conversation into OpenAI-safe chat messages."""
        gpt_messages: List[Dict[str, str]] = []
        for msg in messages or []:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content is None:
                content = ""
            content = str(content)

            if role == "tool":
                gpt_messages.append({"role": "user", "content": f"[ToolResult]\n{content}"})
            elif role in ("system", "user", "assistant"):
                gpt_messages.append({"role": role, "content": content})
            else:
                gpt_messages.append({"role": "user", "content": content})
        return gpt_messages

    def _generate_with_retry(self, conversation, tools, temperature,
                            max_new_tokens, top_p, top_k, min_p,
                            backend: str = "athena", seed=None,
                            cancel_event=None):
        """Generate output with automatic retry on tool call format errors."""
        retry_count = 0
        output_begin_string = None
        prefix_text = ""
        backend = (backend or "athena").strip().lower()

        while retry_count <= self.tool_call_max_retries:
            # If the client already disconnected, a streamed generation will
            # have returned a truncated partial. Don't burn retries re-running
            # tool-call format validation on it — return an EMPTY (not None)
            # output so the caller's Strategy-1 loop breaks cleanly (None would
            # route into Strategy 2's blocking, non-cancel-aware context
            # checkpoint and emit a misleading token-overflow message). The
            # empty output emits nothing and the round loop's next top-of-loop
            # cancel check force-finishes. (No-op when not cancelled.)
            if cancel_event is not None and cancel_event.is_set():
                return "", False
            # Generate
            if backend != "athena":
                conv_for_ext = conversation
                if retry_count > 0:
                    conv_for_ext = list(conversation) + [{
                        "role": "user",
                        "content": "Your previous tool-call formatting was invalid. Output again following the STRICT ATHENA-R1 tool-call protocol."
                    }]
                if backend == "claude":
                    output_str = self.claude_llm_infer(
                        messages=conv_for_ext,
                        temperature=max(0.0, float(temperature)),
                        tools=tools,
                        max_new_tokens=max_new_tokens,
                        top_p=top_p,
                    )
                else:  # gpt (Azure) or gemini — OpenAI-compatible chat
                    output_str = self.gpt_llm_infer(
                        messages=conv_for_ext,
                        temperature=max(0.0, float(temperature)),
                        tools=tools,
                        max_new_tokens=max_new_tokens,
                        top_p=top_p,
                        backend=backend,
                    )
                token_overflow = False
            else:
                output_str, token_overflow = self.llm_infer(
                    messages=conversation,
                    temperature=temperature,
                    tools=tools,
                    output_begin_string=output_begin_string,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    check_token_status=True,
                    top_p=top_p, top_k=top_k, min_p=min_p,
                    cancel_event=cancel_event,
                )

            if output_str is None:
                return output_str, token_overflow

            # If this is a retry, prepend the saved prefix
            if (backend == "athena") and retry_count > 0 and prefix_text:
                full_output = prefix_text + output_str
            else:
                full_output = output_str

            # Truncate </think> repetition: keep first </think> + tool_call if present
            think_close_count = full_output.count("</think>")
            if think_close_count >= 2:
                first_close = full_output.find("</think>")
                after_first = full_output[first_close + len("</think>"):]
                # Check if a <tool_call> follows the first </think>
                tool_match = after_first.find("<tool_call>")
                if tool_match != -1:
                    # Keep everything up to and including the tool_call block
                    tool_end = after_first.find("</tool_call>")
                    if tool_end != -1:
                        full_output = full_output[:first_close + len("</think>")] + after_first[:tool_end + len("</tool_call>")]
                    else:
                        full_output = full_output[:first_close + len("</think>")] + after_first[:tool_match + after_first[tool_match:].find("\n}") + 2]
                else:
                    # No tool_call — keep just up to after first </think>
                    # Find any final answer text before the next </think>
                    second_close = after_first.find("</think>")
                    if second_close != -1:
                        full_output = full_output[:first_close + len("</think>")] + after_first[:second_close].rstrip()
                    # else: only content after first </think>, keep as-is
                logger.info(f"[anti-repetition] Truncated output: {think_close_count} </think> → 1")

            # Validation checks
            if backend != "athena" and "<think>" in full_output:
                if retry_count < self.tool_call_max_retries:
                    retry_count += 1
                    logger.info(f"⚠️  {backend} format error (<think> tags forbidden), retrying ({retry_count}/{self.tool_call_max_retries})...")
                    continue
                return full_output, token_overflow

            # Check for format errors
            if "[TOOL_CALLS]" in full_output and "<tool_call>" not in full_output:
                if retry_count < self.tool_call_max_retries:
                    retry_count += 1
                    logger.info(f"⚠️  Tool call format error, retrying ({retry_count}/{self.tool_call_max_retries})...")
                    continue
                return full_output, token_overflow

            if "<tool_call>" in full_output:
                # Validate format
                tool_call_start = full_output.find("<tool_call>")
                tool_calls_section = full_output[tool_call_start:]

                try:
                    result = self.tooluniverse.extract_function_call_json(
                        tool_calls_section, return_message=False, verbose=False, format=self.format
                    )
                    function_call_json = result[0] if isinstance(result, tuple) else result
                except Exception:
                    function_call_json = None

                if function_call_json is not None:
                    return full_output, token_overflow

                # Format error - retry
                if retry_count < self.tool_call_max_retries:
                    prefix_text = full_output[:tool_call_start] if tool_call_start != -1 else full_output
                    if backend == "athena":
                        output_begin_string = prefix_text
                    retry_count += 1
                    logger.info(f"⚠️  Tool call format error, retrying ({retry_count}/{self.tool_call_max_retries})...")
                    continue

            return full_output, token_overflow

        return full_output, token_overflow

    def get_answer_based_on_unfinished_reasoning(self, conversation, temperature, max_new_tokens,
                                                 outputs=None, return_full_thought=False,
                                                 top_p=0.95, top_k=20, min_p=0.0):
        if conversation[-1]['role'] == 'assistant':
            conversation.append(
                {'role': 'tool', 'content': 'Errors happen during the function call, please come up with the final answer with the current information.'})

        finish_tools_prompt = self.add_special_tools([], call_agent=False, call_agent_level=0)
        finish_tools_prompt = [t for t in finish_tools_prompt if 'Finish' in str(t)]

        final_thought = 'Since I cannot continue reasoning, I will provide the final answer based on the current information and general knowledge.\n\n[FinalAnswer]'
        last_outputs_str = self.llm_infer(messages=conversation,
                                          temperature=temperature,
                                          tools=finish_tools_prompt,
                                          output_begin_string=final_thought,
                                          max_new_tokens=max_new_tokens,
                                          top_p=top_p, top_k=top_k, min_p=min_p)
        if return_full_thought:
            return final_thought + last_outputs_str
        return last_outputs_str

    def report_char_budget(self, max_new_tokens: int = 4096) -> int:
        """Char budget for the report digest: the context window minus room for
        the system prompt/template and the report output, converted chars→tokens."""
        from athena_r1._report import _CHARS_PER_TOKEN
        tokens = max(2000, (self.max_token or 32768) - max_new_tokens - 800)
        return int(tokens * _CHARS_PER_TOKEN)

    def _stream_completion(self, messages, *, temperature=0.7, max_new_tokens=4096,
                           top_p=0.95, top_k=20, min_p=0.0, cancel_event=None):
        """Yield text chunks for a single completion.

        A deliberately self-contained streaming helper (a simpler cousin of
        ``llm_infer`` — fixed presence_penalty, no ``stop`` sequence, no
        INFER_*_PENALTY ablation env overrides) so the run-path ``llm_infer``
        stays untouched. Closing the stream on cancel makes vLLM free the GPU
        immediately."""
        import copy as _copy

        msgs = convert_to_qwen_format(_copy.deepcopy(messages))
        prompt = self.chat_template.render(
            messages=msgs, tools=None, add_generation_prompt=True)
        kwargs = {
            "model": self.model_name,
            "prompt": prompt,
            "top_p": top_p,
            "temperature": temperature,
            "extra_body": {"top_k": top_k, "min_p": min_p},
            "presence_penalty": 1.5,
            "max_tokens": max_new_tokens,
            "stream": True,
        }
        stream = self.model.completions.create(**kwargs)
        # First-chunk watchdog (mirrors llm_infer): without a retry wrapper this
        # path would hang forever if vLLM accepts the request but never emits a
        # first chunk. The timer closes the stream so __next__() raises and the
        # caller sees a fast error instead of an indefinite hang.
        _first_chunk_to = float(os.environ.get(
            "ATHENA_VLLM_FIRST_CHUNK_TIMEOUT", "90"))
        _wd_lock = threading.Lock()
        _wd_closed = [False]

        def _close_stuck_stream(_s=stream, _lock=_wd_lock, _closed=_wd_closed):
            with _lock:
                if not _closed[0]:
                    _closed[0] = True
                    try:
                        _s.close()
                    except Exception:  # noqa: BLE001
                        pass
        _timer = (threading.Timer(_first_chunk_to, _close_stuck_stream)
                  if _first_chunk_to > 0 else None)
        if _timer is not None:
            _timer.daemon = True
            _timer.start()
        try:
            _first = True
            for ch in stream:
                if _first:
                    if _timer is not None:
                        _timer.cancel()
                    _first = False
                if cancel_event is not None and cancel_event.is_set():
                    break
                if getattr(ch, "choices", None):
                    piece = ch.choices[0].text
                    if piece:
                        yield piece
        finally:
            if _timer is not None:
                _timer.cancel()
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    _REPORT_EMPTY_FALLBACK = (
        "I couldn't generate a detailed report from this run's trace. "
        "Please try again or ask a more specific question.")

    def report_snapshots(self, question, digest, *, temperature=0.7,
                         max_new_tokens=4096, top_p=0.95, top_k=20, min_p=0.0,
                         cancel_event=None):
        """Yield progressively-cleaned report snapshots as generation proceeds.

        Each yielded value is the FULL cleaned report-so-far (REPLACE, not
        append) — so a live consumer always shows a well-formed report that
        grows, never the raw <think> draft or a half-cited line. Citation
        validation, repetition trimming, and header normalization are applied
        to every snapshot via clean_report(). The last yield is the complete
        report (or a graceful fallback if generation was empty).

        The agent-trained model treats this out-of-distribution prompt loosely:
        it drafts inside a <think> block and re-emits the report several times,
        so we stop early once it starts repeating (frees the GPU)."""
        from athena_r1._report import (
            REPORT_SYSTEM_PROMPT, clean_report, valid_source_labels)
        # The only citation labels the report may keep — anything else (e.g. a
        # label copied from the prompt's few-shot example) is stripped.
        valid_sources = valid_source_labels(digest)
        # Report generation streams from the local ATHENA-R1 model. API backends
        # (Claude/Gemini) don't load it, so degrade gracefully instead of
        # crashing on a None chat_template/model.
        if self.model is None or self.chat_template is None:
            yield (
                "_Detailed report generation uses the local ATHENA-R1 model, "
                "which isn't loaded for API backends (Claude/Gemini). The "
                "agent's answer is complete; use Backend.ATHENA to generate a "
                "structured clinical report._"
            )
            return
        messages = [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": f"QUESTION:\n{question}\n\n{digest}"},
        ]
        raw = ""
        emitted_at = 0
        last_snap = ""
        for chunk in self._stream_completion(
                messages, temperature=temperature, max_new_tokens=max_new_tokens,
                top_p=top_p, top_k=top_k, min_p=min_p, cancel_event=cancel_event):
            if not chunk:
                continue
            raw += chunk
            # Stop early once the post-think report starts repeating.
            body = raw.split("</think>", 1)[1] if "</think>" in raw else raw
            if body.count("## Recommendation") >= 2:
                break
            # The model often drafts the full report INSIDE a <think> block and
            # then re-writes the real one after </think>. Don't stream the draft
            # — otherwise the live report would grow, then visibly SHRINK when
            # the real (shorter) report starts. Hold live snapshots until the
            # think block closes (or skip entirely if it never opened one).
            in_think_draft = raw.lstrip().startswith("<think>") and "</think>" not in raw
            # Throttle live snapshots to bullet/paragraph boundaries so we don't
            # re-clean+re-render on every token.
            if not in_think_draft and "\n" in chunk and (len(raw) - emitted_at) > 48:
                snap = clean_report(raw, valid_sources)
                if snap.strip() and snap != last_snap:
                    emitted_at = len(raw)
                    last_snap = snap
                    yield snap
        final = clean_report(raw, valid_sources)
        if not final.strip():
            yield self._REPORT_EMPTY_FALLBACK
        elif final != last_snap:
            yield final

    def generate_report(self, question, digest, *, temperature=0.7,
                        max_new_tokens=4096, top_p=0.95, top_k=20, min_p=0.0,
                        cancel_event=None):
        """Single-yield form: the FINAL cleaned report only. Used by the Python
        API and the OpenAI-compat server (where the caller joins the result).
        Internally consumes report_snapshots() and emits just the last one."""
        final = self._REPORT_EMPTY_FALLBACK
        for snap in self.report_snapshots(
                question, digest, temperature=temperature,
                max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k,
                min_p=min_p, cancel_event=cancel_event):
            final = snap
        yield final

    def run_multistep_agent(
        self,
        message: str,
        temperature: float = 0.6,
        max_new_tokens: int = 4096,
        max_round: int = 20,
        call_agent: bool = False,
        call_agent_level: int = 0,
        return_conversation: bool = False,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_retries: int = 10,
        progress_callback: Optional[Any] = None,
        agent_id: str = "main",
        **kwargs: Any,
    ) -> Any:
        """Run Stage-1 multi-step tool reasoning.

        When called recursively from `CallAgentProcessor`, the caller passes
        a child `agent_id` (e.g. "main.sub-1") so that nested streaming events
        carry the right swimlane identifier.
        """
        if len(message) <= 10:
            _msg = "Please provide a valid message with a string longer than 10 characters."
            return (_msg, []) if return_conversation else _msg

        # Reset the CallAgentProcessor's sub-agent counter at the start of
        # each TOP-LEVEL call so sub-agent ids restart at sub-1 per request.
        # The registry is singleton-scoped (lives on the agent instance), so
        # without this reset a long-running server produces visible ids like
        # ``main.sub-237`` after enough traffic. Recursive calls (level > 0)
        # MUST NOT reset — they share the counter with their siblings inside
        # the same run.
        if call_agent_level == 0 and self.tool_processor_registry is not None:
            for proc in self.tool_processor_registry.processors:
                if hasattr(proc, "reset_counter"):
                    proc.reset_counter()

        cancel_event = kwargs.pop("cancel_event", None)
        with AgentRunContext(self, message, call_agent, call_agent_level,
                             progress_callback=progress_callback,
                             agent_id=agent_id) as ctx:
            # Stash on ctx so CallAgentProcessor can propagate it to child
            # recursive runs. Without this, aborting the top-level run
            # leaves sub-agents (and sub-sub-agents) running to completion.
            ctx.cancel_event = cancel_event
            for round_num in range(max_round):
                # Honor cooperative cancellation. Each `agent.answer()` /
                # `answer_streaming()` call passes its OWN `cancel_event`,
                # so concurrent requests on a shared agent instance don't
                # interfere with each other. Older callers can still set
                # the legacy instance flag `_cancel_requested`.
                if (
                    (cancel_event is not None and cancel_event.is_set())
                    or getattr(self, "_cancel_requested", False)
                ):
                    logger.info("Cancellation requested; force-finishing.")
                    forced = ctx.handle_force_finish(
                        temperature, max_new_tokens, top_p, top_k, min_p, return_conversation
                    )
                    # Emit final_answer so streaming clients (AG-UI, OpenAI-
                    # compat) actually deliver the synthesized recovery answer
                    # to the user. Without this, the worker just returns and
                    # the UI shows only the model's last incomplete reasoning
                    # as the "final answer" — losing the engine's recovery
                    # entirely. Mirrors the max_round force-finish emit below.
                    ctx._emit(
                        "final_answer",
                        content=forced if not return_conversation else (forced[0] if forced else None),
                        forced=True,
                    )
                    return forced
                logger.info(f"Round {round_num + 1}/{max_round}")
                ctx._emit("round_start", round=round_num + 1, max_round=max_round)

                # Snapshot conversation length to detect new tool messages added
                # by process_tool_calls in this round.
                prev_conv_len = len(ctx.conversation)
                should_stop, result = ctx.process_tool_calls(
                    temperature, max_new_tokens, top_p, top_k, min_p, return_conversation)
                # Emit any new tool-result messages appended this round.
                for msg in ctx.conversation[prev_conv_len:]:
                    if isinstance(msg, dict) and msg.get("role") in ("tool", "function"):
                        ctx._emit(
                            "tool_result",
                            name=msg.get("name") or msg.get("tool_call_id") or "tool",
                            content=msg.get("content", ""),
                        )
                if should_stop:
                    ctx._emit("final_answer", content=result if not return_conversation else result[0])
                    return result
                # Preemptive Token Validation (Strategy 1)
                current_limit = 0.7
                last_outputs_str = None
                
                # Iteratively lower the safe limit if the model continues to crash
                while current_limit >= 0.3:
                    if current_limit < 0.7:
                        logger.info(f" Model generation crashed or context exhausted. Retrying Strategy 1 with stricter {current_limit:.1f} limit... ")
                        
                    space_ensured = ctx.ensure_context_space(temperature, max_new_tokens, top_p, top_k, min_p, return_conversation, dynamic_limit=current_limit, compact=False)

                    if space_ensured:
                        last_outputs_str = ctx.generate_next_output(
                            temperature, max_new_tokens, top_p, top_k, min_p)
                            
                        # If execution succeeded, break out of loop
                        if last_outputs_str is not None:
                            break
                            
                    current_limit -= 0.2
                        
                # If we've exhausted Strategy 1 loop limits and it STILL crashes, trigger Strategy 2
                if last_outputs_str is None:
                    logger.info(" Model generation crashed at minimum limit. Triggering Strategy 2 (Context Checkpoint). ")
                    ctx.ensure_context_space(temperature, max_new_tokens, top_p, top_k, min_p, return_conversation, compact=True)
                    
                    # Having successfully checkpointed, explicitly generate what fits in the new context
                    last_outputs_str = ctx.generate_next_output(temperature, max_new_tokens, top_p, top_k, min_p)
                    
                    if last_outputs_str is None:
                        # Strategy 2 still couldn't fit the prompt. Rather than
                        # surface a bare "token overflow" error, run the robust
                        # tiered fallback that ALWAYS synthesizes an answer
                        # (hard-trim tool outputs → minimal context → direct
                        # answer). Guarantees the user gets a real response.
                        logger.info(" Strategy 2 overflowed; entering robust emergency answer synthesis. ")
                        answer = ctx._emergency_final_answer(
                            temperature, max_new_tokens, top_p, top_k, min_p)
                        # Emit BOTH events so the UI shows the answer in the
                        # current round (reasoning → .speech) AND in the
                        # final-answer panel (RUN_FINISHED extracts last speech).
                        ctx._emit("reasoning", round=round_num + 1, content=answer)
                        ctx._emit("final_answer", content=answer, forced=True)
                        # MUST match the (response, conversation) tuple shape
                        # callers expect when return_conversation=True —
                        # otherwise `response, conversation = run_multistep_agent(...)`
                        # crashes with "too many values to unpack (expected 2)".
                        return (answer, ctx.conversation) if return_conversation else answer

                    ctx.token_overflow = False

                # Emit the model's reasoning text for this round (with tool
                # calls inline). Strip lengthy whitespace at boundaries.
                if last_outputs_str:
                    ctx._emit("reasoning", round=round_num + 1, content=last_outputs_str)
                    # Detect explicit tool_call blocks so the UI can highlight them.
                    for m in re.finditer(
                        r'<tool_call>(.*?)</tool_call>',
                        last_outputs_str,
                        flags=re.DOTALL,
                    ):
                        raw = m.group(1).strip()
                        tool_name = None
                        try:
                            parsed = json.loads(raw)
                            if isinstance(parsed, dict):
                                tool_name = parsed.get("name")
                        except Exception:  # noqa: BLE001
                            pass
                        ctx._emit(
                            "tool_call",
                            round=round_num + 1,
                            content=raw,
                            tool_name=tool_name,
                        )

                is_final, result = ctx.check_final_answer(last_outputs_str, return_conversation)
                if is_final:
                    ctx._emit(
                        "final_answer",
                        content=result if not return_conversation else result[0],
                    )
                    return result

                ctx.last_outputs.append(last_outputs_str)

            forced = ctx.handle_force_finish(temperature, max_new_tokens, top_p, top_k, min_p, return_conversation)
            # Guarantee a non-empty answer even if force-finish is disabled or
            # its synthesis came back empty (overflow / API hiccup) — fall back
            # to the robust tiered emergency synthesis. The user always gets a
            # real response, never a blank "(no final answer)".
            forced_text = (forced[0] if (return_conversation and isinstance(forced, tuple)) else forced)
            if not (forced_text and str(forced_text).strip()):
                answer = ctx._emergency_final_answer(
                    temperature, max_new_tokens, top_p, top_k, min_p)
                forced = (answer, ctx.conversation) if return_conversation else answer
                forced_text = answer
            ctx._emit("final_answer", content=forced_text, forced=True)
            return forced

    def run_summary_agent(self, thought_calls: str,
                         function_response: str,
                         temperature: float,
                         max_new_tokens: int,
                         top_p=0.95,
                         top_k=20,
                         min_p=0.0) -> str:
        """Summarize tool outputs"""
        logger.info("Summarized Tool Result:")

        base_prompt_template = """
Thought and function calls:
\"\"\"
{thought_calls}
\"\"\"

Function calls' responses:
\"\"\"
{function_response}
\"\"\"

Based on the Thought and function calls, and the function calls' responses, you need to generate a summary of the function calls' responses that fulfills the requirements of the thought. The summary MUST include all necessary information. Directly respond with the summarized summary of the function calls' responses only.
"""

        # Calculate approximate token length of the prompt
        prompt_with_content = base_prompt_template.format(
            thought_calls=thought_calls,
            function_response=function_response
        )

        if self.summary_format_args:
            temperature = self.summary_format_args.get('temperature', temperature)
            max_new_tokens = self.summary_format_args.get('max_new_tokens', max_new_tokens)
            top_p = self.summary_format_args.get('top_p', top_p)
            top_k = self.summary_format_args.get('top_k', top_k)
            min_p = self.summary_format_args.get('min_p', min_p)

        # If prompt length is within limits, process normally.
        # ``self.max_token`` is a TOKEN budget, but ``len(prompt_with_content)``
        # is a CHARACTER count, so we divide by ``compress_ratio`` (chars per
        # token) to compare apples to apples. The old code compared chars
        # directly to tokens — for a 32k-token model that meant prompts of
        # 32k-128k chars (well inside the real budget) needlessly took the
        # expensive chunk-and-merge path.
        estimated_tokens = len(prompt_with_content) / max(self.compress_ratio, 1.0)
        if estimated_tokens <= self.max_token:
            conversation = [{"role": "user", "content": prompt_with_content}]
            logger.info("start standard llm summary inference")

            # Check if summary_format_model exists; use main model if not
            model = self.summary_format_model if self.summary_format_model is not None else self.model
            model_name = self.summary_format_model_name if self.summary_format_model_name is not None else self.model_name

            output = self.llm_infer(messages=conversation,
                                  output_begin_string="<think>\n\n</think>\n\n",
                                  temperature=temperature,
                                  tools=None,
                                  max_new_tokens=max_new_tokens,
                                  top_p=top_p, top_k=top_k, min_p=min_p,
                                  model=model,
                                  model_name=model_name)
            if "</think>" in output:
                output = output.split("</think>")[-1].strip()
            if '[' in output:
                output = output.split('[')[0]
            logger.info("end standard llm summary inference")
            return output

        # If prompt is too long, split function_response into chunks
        logger.info("Prompt too long, splitting into chunks...")

        # Split function_response into chunks that fit within token limit
        # Convert token limit to characters using compress_ratio (chars per token)
        # Reserve ~2000 tokens for prompt template + chat overhead
        chunk_size = int((self.max_token - 2000) * self.compress_ratio * 0.8)
        chunks = [function_response[i:i + chunk_size] for i in range(0, len(function_response), chunk_size)]

        # Process each chunk and collect summaries
        chunk_summaries = []
        model = self.summary_format_model if self.summary_format_model is not None else self.model
        model_name = self.summary_format_model_name if self.summary_format_model_name is not None else self.model_name

        for i, chunk in enumerate(chunks):
            chunk_prompt = base_prompt_template.format(
                thought_calls=thought_calls,
                function_response=chunk
            )

            conversation = [{"role": "user", "content": chunk_prompt}]
            chunk_output = self.llm_infer(messages=conversation,
                                        output_begin_string="<think>\n\n</think>\n\n",
                                        temperature=temperature,
                                        tools=None,
                                        max_new_tokens=max_new_tokens,
                                        top_p=top_p, top_k=top_k, min_p=min_p,
                                        model=model,
                                        model_name=model_name)
            if "</think>" in chunk_output:
                chunk_output = chunk_output.split("</think>")[-1].strip()
            if '[' in chunk_output:
                chunk_output = chunk_output.split('[')[0]
            chunk_summaries.append(chunk_output)

        # If we only have one chunk summary, return it directly
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        # Combine chunk summaries into a final summary
        combined_summary = " ".join(chunk_summaries)
        return combined_summary



    def run_format_agent_full_history(
        self,
        options_str: str,
        conversation_history: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_new_tokens: int = 1024,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
    ) -> str:
        """Native Stage-2 option mapping.

        Appends a user turn to the existing conversation:
            {"role": "user",
             "content": f"Here are the options: {opts}. Based on your previous "
                        f"answer, please select the correct option. Do not use "
                        f"any tools."}
        and continues generation. The model has the FULL agent trajectory
        (system + user_question + all assistant/tool turns) in context.

        The original run_format_agent stripped this and used a fresh standalone
        prompt with only post-</think> text — a distribution shift from training.
        This method preserves the full conversation_history.
        """
        logger.info("start format agent (full-history v2)")
        if not conversation_history:
            logger.info("[full-history v2] no conversation_history provided; cannot match training format")
            return "Error"

        # Build messages list = full history + training-style mapping prompt
        conversation = list(conversation_history)
        conversation.append({
            "role": "user",
            "content": (
                f"Here are the options: {options_str}. Based on your previous "
                f"answer, please select the correct option. Do not use any tools."
            ),
        })

        if self.summary_format_args:
            temperature = self.summary_format_args.get('temperature', temperature)
            max_new_tokens = self.summary_format_args.get('max_new_tokens', max_new_tokens)
            top_p = self.summary_format_args.get('top_p', top_p)
            top_k = self.summary_format_args.get('top_k', top_k)
            min_p = self.summary_format_args.get('min_p', min_p)

        max_retries = 5
        final_extracted_answer = "Error"

        for attempt in range(max_retries):
            logger.info(f"[full-history v2] attempt {attempt + 1}/{max_retries}")
            answer_text = self.llm_infer(messages=conversation,
                                         temperature=temperature,
                                         tools=None,
                                         max_new_tokens=max_new_tokens,
                                         top_p=top_p, top_k=top_k, min_p=min_p,
                                         model=self.summary_format_model,
                                         model_name=self.summary_format_model_name)
            if not answer_text:
                logger.info(f"[full-history v2] empty response, retrying")
                continue

            # Parse: try common letter-extraction patterns
            # 1. "The answer is X" / "X:" prefix / single-letter response
            text = answer_text
            if "</think>" in text:
                text = text.split("</think>")[-1]
            text = text.strip()

            # Direct single letter
            if len(text) >= 1 and text[0].upper() in "ABCDE":
                # Confirm it's a real answer (not just a word starting with a letter)
                if len(text) == 1 or not text[1].isalpha():
                    return text[0].upper()

            # "X: name" pattern
            if len(text) >= 2 and text[1] == ':':
                if text[0].upper() in "ABCDE":
                    return text[0].upper()

            # Search for "The answer is X" / "answer: X" patterns
            for pattern in ["the answer is", "answer is:", "answer:", "final answer:",
                            "the correct answer is", "the final answer is"]:
                idx = text.lower().find(pattern)
                if idx >= 0:
                    tail = text[idx + len(pattern):].strip()
                    # Strip leading punctuation
                    while tail and not tail[0].isalpha():
                        tail = tail[1:]
                    if tail and tail[0].upper() in "ABCDE":
                        return tail[0].upper()

            # Last resort: scan for first standalone A-E letter
            import re as _re
            m = _re.search(r'\b([A-E])\b', text)
            if m:
                return m.group(1)

            logger.info(f"[full-history v2] could not parse letter from: {text[:200]!r}")
            final_extracted_answer = text[:32]

        return final_extracted_answer

    def run_gpt_format_agent(
        self,
        conversation_history: List[Dict[str, Any]],
        options_str: str,
        temperature: float = 1.0,
        max_new_tokens: int = 8192,
        message: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> str:
        """GPT-based Stage-2 letter mapper. Reads the full agent conversation
        plus an options string ("A. ..., B. ..., C. ...") and asks GPT-5 to
        pick a single letter.

        For backward compat the legacy ``message`` + ``answer`` final-answer-
        only mode is still supported (when ``conversation_history`` is empty),
        but the modern caller should always pass the full conversation.
        """
        logger.info("start format agent")

        # Fast path: the agent already emitted an explicit letter.
        if answer:
            possible = (
                answer.split("[FinalAnswer]")[-1] if "[FinalAnswer]" in answer
                else (answer.split("\n\n")[-1] if "\n\n" in answer else answer.strip())
            )
            if len(possible) == 1 and possible in "ABCDE":
                return possible
            if len(possible) > 1 and possible[1] == ":" and possible[0] in "ABCDE":
                logger.debug(f"choice: {possible[0]}")
                return possible[0]

        if conversation_history:
            # Full-history mode: build the GPT prompt from the full reasoning
            # trace, then ask GPT to select the option.
            conversation = self._build_gpt_mapping_from_history(
                conversation_history, options_str
            )
            logger.info(
                f"[gpt_format_agent] full-history mode, {len(conversation)} messages"
            )
        else:
            # Legacy mode: question + final answer only. Kept for back-compat
            # with callers that don't have access to the full conversation.
            conversation = self.set_system_prompt(
                [],
                "You are helpful assistant to transform the answer of agent to "
                "the final answer of 'A', 'B', 'C', 'D'.",
            )
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        f"{message or ''}\nOptions: {options_str}\n"
                        f"The final answer of agent: {answer or ''}\n"
                        "The answer is (must be a letter):"
                    ),
                }
            )

        # Retry on transient Azure errors (APIConnectionError, APITimeoutError, 5xx).
        # Backoff: 2, 4, 8, 16 s — total worst-case ~30 s before giving up.
        import time as _time
        last_err = None
        for attempt in range(5):
            try:
                responses = self.gpt_client.chat.completions.create(
                    model="gpt-5",
                    messages=conversation,
                    temperature=1.0,
                    max_completion_tokens=max_new_tokens,
                    timeout=60,
                )
                answer = responses.choices[0].message.content
                logger.debug(f"GPT5 reply: {answer[:200] if answer else ''}")
                return answer.strip() if answer else None
            except Exception as e:
                last_err = e
                etype = type(e).__name__
                if etype in ("APIConnectionError", "APITimeoutError", "RateLimitError",
                             "InternalServerError", "APIError"):
                    sleep_s = 2 * (2 ** attempt)
                    logger.info(f"[run_gpt_format_agent] {etype}, retry {attempt+1}/5 after {sleep_s}s")
                    _time.sleep(sleep_s)
                    continue
                logger.info(f"[run_gpt_format_agent] GPT call failed: {etype}: {e}")
                return None
        logger.info(f"[run_gpt_format_agent] all 5 retries exhausted: {type(last_err).__name__}: {last_err}")
        return None

    def _build_gpt_mapping_from_history(self, conversation_history, options_message):
        """Build GPT mapping prompt from the full agent conversation history.

        The full reasoning trace is presented as context, then a mapping prompt
        asks GPT to select the option.
        """
        gpt_messages = []
        gpt_messages.append({
            "role": "system",
            "content": "You are a helpful assistant. You will be shown a medical agent's "
                       "full reasoning trace including its thoughts, tool calls, and tool results. "
                       "Based on this trace, select the correct multiple choice answer."
        })

        # Build the full trace from conversation history
        trace_parts = []
        for turn in conversation_history:
            role = turn.get('role', '')
            content = turn.get('content', '')
            if role == 'system':
                continue  # skip system prompts from the agent
            elif role == 'user':
                trace_parts.append(f"[User]: {content}")
            elif role == 'assistant':
                trace_parts.append(f"[Assistant]: {content}")
            elif role == 'tool':
                # Truncate very long tool outputs to avoid hitting token limits
                if len(content) > 2000:
                    content = content[:2000] + "... [truncated]"
                trace_parts.append(f"[Tool Result]: {content}")

        full_trace = "\n".join(trace_parts)

        gpt_messages.append({
            "role": "user",
            "content": f"Here is the agent's full reasoning trace:\n\n{full_trace}\n\n"
                       f"{options_message}\n"
                       f"Based on the agent's reasoning above, select the correct option. "
                       f"The answer is (must be a letter):"
        })

        return gpt_messages



    def run_context_summary_agent(self, conversation: list,
                          temperature: float,
                          max_new_tokens: int,
                          top_p=0.95,
                          top_k=20,
                          min_p=0.0) -> str:
        """Run context summary agent"""
        logger.info("start context summary agent")

        # Helper to format conversation
        def format_conv(conv):
            formatted = ""
            for msg in conv:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    continue
                formatted += f"{role.upper()}: {content}\n\n"
            return formatted.replace('"""', "\\\"\\\"\\\"")

        formatted_conversation = format_conv(conversation)

        # Check length
        # Check length. 1 token ~= 3.5 chars. We want to stay well below 32k tokens.
        SAFE_CHAR_LIMIT = 80000 
        HARD_CHAR_LIMIT = 100000

        if len(formatted_conversation) > SAFE_CHAR_LIMIT:
            logger.info(f"Context too long ({len(formatted_conversation)} chars), attempting to compress...")

            # Compress by summarizing tool results
            compressed_conversation = []
            last_thought = ""

            for msg in conversation:
                msg_copy = msg.copy()
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "assistant":
                    last_thought = content

                if role == "tool" and len(content) > 3000:
                     logger.info(f"Summarizing tool output of length {len(content)}...")
                     summary = self.run_summary_agent(
                         thought_calls=last_thought,
                         function_response=content,
                         temperature=0.1,
                         max_new_tokens=10240
                     )
                     msg_copy['content'] = f"[Summarized Tool Output]: {summary}"

                compressed_conversation.append(msg_copy)

            formatted_conversation = format_conv(compressed_conversation)

            if len(formatted_conversation) > SAFE_CHAR_LIMIT:
                 logger.info(f"Still too long ({len(formatted_conversation)} chars), chunking linearly by rounds...")
                 
                 # Group messages into complete semantic rounds
                 rounds = []
                 current_round = []
                 for msg in compressed_conversation:
                     if msg.get('role') == 'user' and current_round:
                         rounds.append(current_round)
                         current_round = [msg]
                     else:
                         current_round.append(msg)
                 if current_round:
                     rounds.append(current_round)

                 # Pack rounds into chunks that fit inside SAFE_CHAR_LIMIT
                 chunks = []
                 current_chunk = []
                 current_len = 0
                 
                 for rnd in rounds:
                     rnd_str = format_conv(rnd)
                     if current_len + len(rnd_str) > SAFE_CHAR_LIMIT and current_chunk:
                         chunks.append(current_chunk)
                         current_chunk = rnd
                         current_len = len(rnd_str)
                     else:
                         current_chunk.extend(rnd)
                         current_len += len(rnd_str)
                         
                 if current_chunk:
                     chunks.append(current_chunk)
                     
                 # If we somehow have 1 massive round that is larger than the limit, forcefully cut it
                 if len(chunks) == 1:
                     chunk1 = compressed_conversation[:len(compressed_conversation)//2]
                     chunk2 = compressed_conversation[len(compressed_conversation)//2:]
                     chunks = [chunk1, chunk2]

                 # Summarize each chunk
                 summaries = []
                 for i, chunk in enumerate(chunks):
                     logger.info(f"Summarizing chunk {i+1} of {len(chunks)}...")
                     summaries.append(self.run_context_summary_agent(
                         chunk, temperature, max_new_tokens, top_p, top_k, min_p
                     ))

                 # Merge all summaries together
                 new_conv = [{"role": "user", "content": "Please merge the following chronological summaries into one master summary."}]
                 for summ in summaries:
                     new_conv.append({"role": "assistant", "content": summ})
                 
                 formatted_conversation = format_conv(new_conv)

        prompt_content = f"""Here is the conversation so far:
\"\"\"
{formatted_conversation}
\"\"\"

Now, please follow the instructions to create the handoff summary.
"""

        summary_conversation = []
        summary_conversation = self.set_system_prompt(summary_conversation, self.context_summary_prompt)
        summary_conversation.append({"role": "user", "content": prompt_content})

        if self.summary_format_args:
             temperature = self.summary_format_args.get('temperature', temperature)
             max_new_tokens = self.summary_format_args.get('max_new_tokens', max_new_tokens)
             top_p = self.summary_format_args.get('top_p', top_p)
             top_k = self.summary_format_args.get('top_k', top_k)
             min_p = self.summary_format_args.get('min_p', min_p)

        summary = self.llm_infer(messages=summary_conversation,
                              temperature=temperature,
                              tools=None,
                              max_new_tokens=max_new_tokens,
                              top_p=top_p, top_k=top_k, min_p=min_p,
                              model=self.summary_format_model,
                              model_name=self.summary_format_model_name)

        if "</think>" in summary:
            summary = summary.split("</think>")[-1].strip()

        logger.info(f"Generated Context Summary:\n{summary}")
        return summary