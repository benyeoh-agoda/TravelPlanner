"""
PlannerR1Agent: chat-loop agent for --prompt_style planner_r1 / planner_r1_json.

Implements the Planner-R1 design (arXiv:2509.25779):
- System prompt contains only behavior guidelines and the final-answer format.
- Tools are exposed via OpenAI native tool-calling (bind_tools), not by listing
  them in the prompt as plain text.
- One LLM call per turn; the model either emits a tool call or the final answer.
- Tool results come back as ToolMessage objects (proper chat history).
- Final answer is detected by <answer>...</answer> (planner_r1_json) or
  a 'Day 1:' block (planner_r1).

The original ReactAgent and its step() loop are completely untouched.
"""

import re
import os
import sys
import json
import time
import importlib
import tiktoken
import openai

from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import base as _lc_openai_base
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain 0.1.x silently drops `reasoning`/`reasoning_content` returned by
# vLLM reasoning models. Patch both directions so traces round-trip in history.
# No-op for OpenAI endpoints — they never return a `reasoning` field.
_orig_msg_to_dict = _lc_openai_base._convert_message_to_dict
_orig_dict_to_msg = _lc_openai_base._convert_dict_to_message


def _patched_convert_message_to_dict(message):
    d = _orig_msg_to_dict(message)
    ak = getattr(message, "additional_kwargs", {}) or {}
    reasoning = ak.get("reasoning_content") or ak.get("reasoning")
    if reasoning:
        d["reasoning_content"] = reasoning
    return d


def _patched_convert_dict_to_message(d):
    msg = _orig_dict_to_msg(d)
    if isinstance(msg, AIMessage):
        reasoning = d.get("reasoning_content") or d.get("reasoning")
        if reasoning:
            msg.additional_kwargs.setdefault("reasoning_content", reasoning)
    return msg


_lc_openai_base._convert_message_to_dict = _patched_convert_message_to_dict
_lc_openai_base._convert_dict_to_message = _patched_convert_dict_to_message

from prompts import PLANNER_R1_SYSTEM_PROMPT, PLANNER_R1_JSON_SYSTEM_PROMPT
from tool_schemas import TOOL_SCHEMAS
from tool_executor import execute_tool

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')


def _catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.RateLimitError:
        print("RateLimitError")
        time.sleep(60)
    elif error == openai.BadRequestError:
        print("BadRequestError")
        raise sys.exc_info()[1]
    elif error == openai.APIError:
        print("APIError")
    elif error == openai.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)


class PlannerR1Agent:
    def __init__(self,
                 prompt_style: str = 'planner_r1',
                 tools: List[str] = None,
                 react_llm_name: str = 'gpt-3.5-turbo-1106',
                 planner_llm_name: str = 'gpt-3.5-turbo-1106',
                 max_steps: int = 30,
                 max_tokens: int = None,
                 city_file_path: str = '../database/background/citySet.txt',
                 base_url: str = None,
                 no_think: bool = False,
                 reasoning_effort: str = None,
                 ) -> None:

        self.prompt_style = prompt_style
        self.max_steps = max_steps
        self.system_prompt = (
            PLANNER_R1_JSON_SYSTEM_PROMPT
            if prompt_style == 'planner_r1_json'
            else PLANNER_R1_SYSTEM_PROMPT
        )

        # Build LLM client with native tool-calling.
        # Mirror the model dispatch from ReactAgent, but use bind_tools() so
        # the gateway exposes the tool schemas via the chat API.
        if react_llm_name.startswith('ollama:'):
            ollama_model = react_llm_name.split(":", 1)[1] or "llama3"
            self.max_token_length = 30000
            base_llm = ChatOllama(model=ollama_model, temperature=0)
        elif react_llm_name in ['gemini']:
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required for 'gemini'.")
            base_llm = ChatGoogleGenerativeAI(
                temperature=0, model="gemini-pro", google_api_key=GOOGLE_API_KEY)
            self.max_token_length = 30000
        else:
            self.max_token_length = 30000
            extra_kwargs = {}
            if no_think:
                extra_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            llm_kwargs = dict(
                model_name=react_llm_name,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=base_url or os.environ.get('OPENAI_API_BASE'),
                request_timeout=30,
                max_retries=3,
                model_kwargs=extra_kwargs,
            )
            if reasoning_effort is not None:
                llm_kwargs['reasoning_effort'] = reasoning_effort
            if max_tokens is not None:
                llm_kwargs['max_tokens'] = max_tokens
            base_llm = ChatOpenAI(**llm_kwargs)

        self.llm = base_llm.bind_tools(TOOL_SCHEMAS)

        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.tools = self._load_tools(tools or [], planner_model_name=planner_llm_name)
        self.city_set = self._load_city(city_file_path)
        self.retry_record: Dict[str, int] = {key: 0 for key in self.tools}
        self.retry_record['invalidAction'] = 0

        self.answer = ''
        self.finished = False
        self.current_data = None

    def run(self, query: str) -> tuple:
        """
        Run the chat loop for a single query.

        Returns
        -------
        (answer_str, transcript_str, action_log)
          answer_str    : final plan text (empty string if the agent did not finish)
          transcript_str: human-readable multi-turn transcript for _two-stage_results_logs
          action_log    : list of {step, tool, args, observation} dicts
        """
        self.answer = ''
        self.finished = False
        self.current_data = None
        self.retry_record = {k: 0 for k in self.retry_record}
        if 'notebook' in self.tools:
            self.tools['notebook'].reset()

        action_log = []
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=query)]
        nudged = False  # allow at most one empty-response nudge

        for step_n in range(self.max_steps):
            if self._token_count(messages) > self.max_token_length:
                print(f"[PlannerR1Agent] Token limit reached at step {step_n}.")
                break

            # LLM call
            ai_msg = self.llm.invoke(messages)
            print(f"[PlannerR1Agent] Step {step_n} response: content={repr(ai_msg.content[:200] if ai_msg.content else '')}, tool_calls={[c['name'] for c in (getattr(ai_msg,'tool_calls',None) or [])]}, metadata={ai_msg.response_metadata}", flush=True)

            # In no_think mode vLLM prefixes generation with <think>\n\n</think>\n\n
            # but returns no reasoning_content. The Qwen3 chat template only
            # includes that prefix for history turns when reasoning_content is
            # truthy, so we inject an empty string to keep history consistent.
            messages.append(ai_msg)

            # When thinking mode is on, vLLM puts the answer in content and the
            # chain-of-thought in reasoning. If content is empty but reasoning is
            # present, the model ran out of max_tokens mid-thought — treat it as
            # an empty response so the nudge logic fires.
            content = ai_msg.content or ''
            if not content:
                reasoning = (ai_msg.additional_kwargs.get('reasoning') or
                             ai_msg.response_metadata.get('reasoning') or '')
                if reasoning:
                    print(f"[PlannerR1Agent] Step {step_n}: content empty, reasoning truncated "
                          f"({len(reasoning)} chars) — treating as empty response.")

            # Check for final answer in text content
            final = self._detect_final_answer(content)
            if final is not None:
                self.answer = final
                self.finished = True
                print(f"[PlannerR1Agent] Final answer detected at step {step_n}.")
                return self.answer, self._serialize(messages), action_log

            # Dispatch native tool calls
            tool_calls = getattr(ai_msg, 'tool_calls', None) or []
            if not tool_calls:
                if nudged:
                    print("[PlannerR1Agent] No tool call or final answer after nudge; stopping.")
                    break
                messages.append(HumanMessage(
                    content="Please call a tool to gather more information, "
                            "or emit your final travel plan."))
                nudged = True
                continue

            nudged = False
            for call in tool_calls:
                tool_name = call['name']
                tool_args = call.get('args', {})
                call_id   = call.get('id', f'call_{step_n}')

                print(f"[PlannerR1Agent] Step {step_n}: {tool_name}({tool_args})")

                obs, self.current_data, is_terminal = execute_tool(
                    tool_name, tool_args,
                    self.tools, self.city_set,
                    self.retry_record, self.current_data,
                )

                print(f"[PlannerR1Agent] Observation: {obs[:120]}{'...' if len(obs) > 120 else ''}")

                messages.append(ToolMessage(content=obs, tool_call_id=call_id))
                action_log.append({
                    "step": step_n,
                    "tool": tool_name,
                    "args": tool_args,
                    "observation": obs,
                })

                if is_terminal:
                    # Planner tool was called — treat its output as the answer
                    self.answer = obs
                    self.finished = True
                    return self.answer, self._serialize(messages), action_log

        return self.answer, self._serialize(messages), action_log

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_final_answer(self, text: str):
        """Return text if it contains a final answer, else None."""
        if not text:
            return None
        if self.prompt_style == 'planner_r1_json':
            if re.search(r'<answer>.*?</answer>', text, re.DOTALL):
                return text
            return None
        # planner_r1 (Option B): 'Day 1:' header present
        if re.search(r'\bDay\s*1\s*:', text):
            return text
        return None

    def _token_count(self, messages) -> int:
        total = 0
        for m in messages:
            content = getattr(m, 'content', '') or ''
            total += len(self.enc.encode(str(content)))
        return total

    def _serialize(self, messages) -> str:
        """Convert the message list to a readable transcript string."""
        lines = []
        for m in messages:
            content = getattr(m, 'content', '') or ''
            tool_calls = getattr(m, 'tool_calls', None)
            if isinstance(m, SystemMessage):
                lines.append(f"[System]\n{content}")
            elif isinstance(m, HumanMessage):
                lines.append(f"[User]\n{content}")
            elif isinstance(m, AIMessage):
                if tool_calls:
                    calls_str = '; '.join(
                        f"{c['name']}({json.dumps(c.get('args', {}))})"
                        for c in tool_calls
                    )
                    lines.append(f"[Assistant]\n<tool_call> {calls_str}")
                    if content:
                        lines[-1] += f"\n{content}"
                else:
                    lines.append(f"[Assistant]\n{content}")
            elif isinstance(m, ToolMessage):
                lines.append(f"[Tool Result]\n{content}")
        return '\n\n'.join(lines)

    def _load_tools(self, tools: List[str], planner_model_name=None) -> Dict[str, Any]:
        tools_map = {}
        for tool_name in tools:
            module = importlib.import_module("tools.{}.apis".format(tool_name))
            if tool_name == 'planner' and planner_model_name is not None:
                tools_map[tool_name] = getattr(
                    module, tool_name[0].upper() + tool_name[1:]
                )(model_name=planner_model_name)
            else:
                tools_map[tool_name] = getattr(
                    module, tool_name[0].upper() + tool_name[1:]
                )()
        return tools_map

    def _load_city(self, city_set_path: str) -> List[str]:
        try:
            return open(city_set_path, 'r').read().strip().split('\n')
        except FileNotFoundError:
            print(f"[PlannerR1Agent] Warning: city file not found at {city_set_path}")
            return []
