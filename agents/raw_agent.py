"""
raw-agent.py
============
A minimal CrewAI-like multi-agent framework.

Supported LLM providers
-----------------------
- Google Gemini  (default)  — set llm="gemini-2.5-flash" or any gemini-* model
- Groq                      — set llm="llama-3.3-70b-versatile" or any groq model

The provider is detected automatically from the model name, or you can pass
`provider="gemini"` / `provider="groq"` explicitly.

Core classes
------------
- Tool        : A callable capability an Agent can use.
- Agent       : An LLM-powered actor that can use tools (ReAct-style loop).
- Task        : A unit of work assigned to an Agent, with optional context chaining.
- RawAgent    : Orchestrator that runs a list of Tasks sequentially or in parallel.
"""

from __future__ import annotations

import os
import json
import time
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

# Groq model name fragments — used for auto-detection
_GROQ_MODEL_PREFIXES = (
    "llama",
    "mixtral",
    "gemma",
    "whisper",
    "qwen",
    "deepseek",
    "distil",
)


def _detect_provider(model: str) -> str:
    """
    Infer the LLM provider from the model name.

    Returns "gemini" or "groq".
    """
    lower = model.lower()
    if lower.startswith("gemini"):
        return "gemini"
    if any(lower.startswith(p) for p in _GROQ_MODEL_PREFIXES):
        return "groq"
    # Default to gemini for unknown model names
    return "gemini"


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class Tool:
    """
    A named, callable capability that an Agent can invoke.

    Parameters
    ----------
    name : str
        Short slug used to identify the tool (e.g. "web_search").
    description : str
        Human-readable description telling the LLM *when* and *how* to use
        this tool, including what arguments to pass.
    func : Callable[..., str]
        The actual Python function to call.  It must accept keyword arguments
        and return a plain string result.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., str],
    ) -> None:
        self.name = name
        self.description = description
        self.func = func

    def __call__(self, **kwargs: Any) -> str:
        return self.func(**kwargs)

    def to_schema(self) -> Dict[str, Any]:
        """Return a JSON-serialisable description for prompt injection."""
        return {"name": self.name, "description": self.description}

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


# ---------------------------------------------------------------------------
# Internal Pydantic models used by Agent._call_llm
# ---------------------------------------------------------------------------

class _ToolCall(BaseModel):
    tool_name: str
    # NOTE: Gemini API does not support Dict[str, Any] (additionalProperties).
    # We receive arguments as a JSON-encoded string and parse it ourselves.
    arguments_json: str  # e.g. '{"query": "AI agents"}'


class _AgentResponse(BaseModel):
    """
    Schema for a single LLM reasoning step.

    Fields
    ------
    thought      : The agent's internal reasoning.
    action       : Either "use_tool" or "final_answer".
    tool_call    : Populated when action == "use_tool".
    final_answer : Populated when action == "final_answer".
    """
    thought: str
    action: str                          # "use_tool" | "final_answer"
    tool_call: Optional[_ToolCall] = None
    final_answer: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    An LLM-powered actor that optionally has access to a set of Tools.

    The Agent follows a ReAct (Reason + Act) loop:
      1. Build a prompt with the current context + tool results so far.
      2. Ask the LLM which action to take next (structured JSON output).
      3. If the LLM says "use_tool"  → call the tool, append result, repeat.
      4. If the LLM says "final_answer" → return the answer.
    The loop is capped at `max_iterations` steps to prevent infinite loops.

    Parameters
    ----------
    role          : Short role title (e.g. "Research Analyst").
    goal          : What this agent is trying to achieve.
    backstory     : Additional personality / context injected into every prompt.
    llm           : Gemini model id (default: "gemini-2.0-flash").
    tools         : List of Tool objects available to this agent.
    max_iterations: Maximum ReAct loop iterations before forcing a final answer.
    verbose       : Whether to print step-by-step reasoning to stdout.
    """

    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str = "You are a helpful assistant.",
        llm: str = "gemini-2.5-flash",
        provider: Optional[str] = None,   # "gemini" | "groq" | None (auto-detect)
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 5,
        max_retries: int = 3,
        verbose: bool = True,
    ) -> None:
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm = llm
        self.provider: str = provider or _detect_provider(llm)
        self.tools: List[Tool] = tools or []
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.verbose = verbose

        # Build a quick name→Tool lookup
        self._tool_map: Dict[str, Tool] = {t.name: t for t in self.tools}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        if self.tools:
            schemas = json.dumps([t.to_schema() for t in self.tools], indent=2)
            tool_section = f"""
                You have access to the following tools:
                {schemas}

                When you need to use a tool, respond with:
                "action": "use_tool"
                "tool_call": {{"tool_name": "<name>", "arguments_json": "{{\\"key\\": \\"value\\"}}"}}

                IMPORTANT: arguments_json MUST be a valid JSON string (double-quoted, escaped).
                Example: arguments_json = "{{\\"query\\": \\"AI agents\\"}}"

                When you have a final answer, respond with:
                "action": "final_answer"
                "final_answer": "<your complete answer>"
                """
        else:
            tool_section = (
                'You have no tools.  Always set "action": "final_answer".'
            )

        return f"""You are a {self.role}.
                Backstory: {self.backstory}
                Goal: {self.goal}

                {tool_section}

                Always respond with a single JSON object with these fields:
                thought      (string)  : your internal reasoning
                action       (string)  : "use_tool" or "final_answer"
                tool_call    (object|null) : {{"tool_name": "...", "arguments_json": "..."}}
                final_answer (string|null) : your complete answer when done
                """

    def _call_llm(self, messages: List[Dict[str, str]]) -> _AgentResponse:
        """Route to the correct provider backend."""
        if self.provider == "groq":
            return self._call_groq(messages)
        return self._call_gemini(messages)

    # ------------------------------------------------------------------
    # Gemini backend
    # ------------------------------------------------------------------

    def _call_gemini(self, messages: List[Dict[str, str]]) -> _AgentResponse:
        """Call Google Gemini with structured JSON output.
        Retries up to self.max_retries times on 429 quota errors.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set."
            )

        client = genai.Client(api_key=api_key)

        # Flatten message history into a single prompt string
        full_prompt = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}" for m in messages
        )

        last_exc: Exception = RuntimeError("No attempts made.")
        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.llm,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=_AgentResponse,
                        temperature=0.3,
                        system_instruction=self._build_system_prompt(),
                    ),
                )
                return response.parsed  # type: ignore[return-value]
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = 20 * attempt  # 20s, 40s, 60s …
                    self._log(
                        "RETRY",
                        f"[Gemini] Rate-limited (429). Waiting {wait}s before retry "
                        f"{attempt}/{self.max_retries}...",
                    )
                    time.sleep(wait)
                else:
                    raise
        raise last_exc

    # ------------------------------------------------------------------
    # Groq backend
    # ------------------------------------------------------------------

    def _call_groq(self, messages: List[Dict[str, str]]) -> _AgentResponse:
        """Call Groq (OpenAI-compatible) with JSON mode.
        Parses the raw JSON string into _AgentResponse manually.
        Retries up to self.max_retries times on 429 / rate-limit errors.
        """
        try:
            from groq import Groq  # lazy import — only required when using Groq
        except ImportError:
            raise ImportError(
                "The 'groq' package is required for Groq models. "
                "Install it with:  pip install groq"
            )

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set."
            )

        client = Groq(api_key=api_key)
        system_prompt = self._build_system_prompt()

        # Groq uses the OpenAI chat format — build the message list properly
        groq_messages = [{"role": "system", "content": system_prompt}] + messages

        last_exc: Exception = RuntimeError("No attempts made.")
        for attempt in range(1, self.max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model=self.llm,
                    messages=groq_messages,  # type: ignore[arg-type]
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                raw_json = completion.choices[0].message.content or "{}"
                data = json.loads(raw_json)

                # Normalise tool_call sub-object
                tool_call_data = data.get("tool_call")
                tool_call = None
                if tool_call_data and isinstance(tool_call_data, dict):
                    # Groq may return arguments as a dict — re-serialise to JSON string
                    args = tool_call_data.get("arguments_json", "{}")
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    tool_call = _ToolCall(
                        tool_name=tool_call_data.get("tool_name", ""),
                        arguments_json=args,
                    )

                return _AgentResponse(
                    thought=data.get("thought", ""),
                    action=data.get("action", "final_answer"),
                    tool_call=tool_call,
                    final_answer=data.get("final_answer"),
                )
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    wait = 10 * attempt  # 10s, 20s, 30s …
                    self._log(
                        "RETRY",
                        f"[Groq] Rate-limited. Waiting {wait}s before retry "
                        f"{attempt}/{self.max_retries}...",
                    )
                    time.sleep(wait)
                else:
                    raise
        raise last_exc

    def _log(self, tag: str, message: str) -> None:
        if self.verbose:
            print(f"\n[{self.role}][{tag}] {message}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, task_description: str, context: str = "") -> str:
        """
        Execute the ReAct loop for a given task description.

        Parameters
        ----------
        task_description : What the agent needs to accomplish.
        context          : Optional prior context (e.g. output of a previous task).

        Returns
        -------
        str : The agent's final answer.
        """
        messages: List[Dict[str, str]] = []

        # Initial user message
        user_content = task_description
        if context:
            user_content = f"Context from previous steps:\n{context}\n\n{task_description}"
        messages.append({"role": "user", "content": user_content})

        self._log("START", f"Task: {task_description[:120]}")

        for iteration in range(1, self.max_iterations + 1):
            self._log("ITER", f"Iteration {iteration}/{self.max_iterations}")

            step: _AgentResponse = self._call_llm(messages)

            self._log("THOUGHT", step.thought)

            if step.action == "final_answer":
                if step.final_answer:
                    self._log("DONE", step.final_answer[:200])
                    return step.final_answer
                else:
                    self._log("WARN", "LLM returned final_answer action but no text.")
                    break

            elif step.action == "use_tool":
                if not step.tool_call:
                    self._log("WARN", "LLM chose use_tool but gave no tool_call.")
                    break

                tool_name = step.tool_call.tool_name

                # Parse arguments_json safely
                try:
                    arguments = json.loads(step.tool_call.arguments_json or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                if tool_name not in self._tool_map:
                    tool_result = f"Error: tool '{tool_name}' not found."
                    self._log("TOOL_ERR", tool_result)
                else:
                    self._log("TOOL", f"Calling '{tool_name}' with {arguments}")
                    try:
                        tool_result = self._tool_map[tool_name](**arguments)
                    except Exception as exc:
                        tool_result = f"Tool error: {exc}"
                    self._log("TOOL_RESULT", str(tool_result)[:300])

                # Append the assistant turn + tool observation to history
                messages.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "thought": step.thought,
                        "action": "use_tool",
                        "tool_call": {
                            "tool_name": tool_name,
                            "arguments_json": json.dumps(arguments),
                        },
                    }),
                })
                messages.append({
                    "role": "user",
                    "content": f"Tool result for '{tool_name}':\n{tool_result}",
                })

            else:
                self._log("WARN", f"Unknown action '{step.action}'. Stopping.")
                break

        # Fallback: ask LLM for a final answer with whatever it knows
        self._log("FALLBACK", "Max iterations reached — requesting final answer.")
        messages.append({
            "role": "user",
            "content": (
                "You have reached the maximum number of iterations. "
                "Please provide your best final answer now."
            ),
        })
        step = self._call_llm(messages)
        return step.final_answer or "Agent could not produce a final answer."


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class Task:
    """
    A unit of work that wraps a description, an expected output format,
    and an optional Agent assignment.

    Parameters
    ----------
    description     : What needs to be done.
    expected_output : Description of what a good result looks like.
    agent           : The Agent responsible for this task.
    context_tasks   : Other Task objects whose output feeds into this task.
    """

    def __init__(
        self,
        description: str,
        expected_output: str,
        agent: Optional[Agent] = None,
        context_tasks: Optional[List["Task"]] = None,
        output_file: Optional[str] = None,
    ) -> None:
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.context_tasks: List["Task"] = context_tasks or []
        self.output_file = output_file

        # Populated after run()
        self.output: Optional[str] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_task_prompt(self) -> str:
        return (
            f"Task:\n{self.description}\n\n"
            f"Expected Output:\n{self.expected_output}"
        )

    def _build_context(self) -> str:
        parts = []
        for t in self.context_tasks:
            if t.output:
                parts.append(
                    f"--- Output from task: {t.description[:60]}... ---\n{t.output}"
                )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> str:
        """
        Execute this task using its assigned Agent.

        Returns
        -------
        str : The agent's output, also stored in self.output.
        """
        if self.agent is None:
            raise ValueError(
                f"Task '{self.description[:60]}' has no agent assigned."
            )

        context = self._build_context()
        prompt = self._build_task_prompt()
        self.output = self.agent.run(task_description=prompt, context=context)

        if self.output and self.output_file:
            try:
                with open(self.output_file, "w", encoding="utf-8") as f:
                    f.write(self.output)
            except Exception as e:
                print(f"Error saving output to {self.output_file}: {e}")

        return self.output

    def __repr__(self) -> str:
        status = "done" if self.output else "pending"
        return f"Task(description={self.description[:40]!r}, status={status!r})"


# ---------------------------------------------------------------------------
# RawAgent  (the Crew / Orchestrator)
# ---------------------------------------------------------------------------

class RawAgent:
    """
    Orchestrates a list of Tasks across a set of Agents.

    Supported process modes
    -----------------------
    "sequential"  : Tasks run one after another.  Each task can reference
                    previous tasks via `context_tasks` for chaining.
    "parallel"    : All tasks run concurrently using a thread pool.
                    Context chaining is not available in this mode.

    Parameters
    ----------
    tasks   : Ordered list of Task objects to execute.
    agents  : List of Agents (informational; agents are embedded in tasks).
    process : "sequential" (default) or "parallel".
    verbose : Whether to print orchestration logs.
    """

    def __init__(
        self,
        tasks: List[Task],
        agents: List[Agent],
        process: str = "sequential",
        verbose: bool = True,
    ) -> None:
        self.tasks = tasks
        self.agents = agents
        self.process = process
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"\n[RawAgent] {message}")

    def run(self) -> List[str]:
        """
        Run all tasks and return their outputs as a list of strings.

        Returns
        -------
        List[str] : Outputs in the same order as self.tasks.
        """
        if self.process == "sequential":
            return self._run_sequential()
        elif self.process == "parallel":
            return self._run_parallel()
        else:
            raise ValueError(
                f"Unknown process type '{self.process}'. "
                "Choose 'sequential' or 'parallel'."
            )

    def _run_sequential(self) -> List[str]:
        self._log(f"Starting SEQUENTIAL run of {len(self.tasks)} task(s).")
        outputs: List[str] = []
        for i, task in enumerate(self.tasks, 1):
            self._log(
                f"Running task {i}/{len(self.tasks)}: "
                f"{task.description[:60]}..."
            )
            result = task.run()
            outputs.append(result)
            self._log(f"Task {i} complete.")
        self._log("All tasks finished.")
        return outputs

    def _run_parallel(self) -> List[str]:
        self._log(f"Starting PARALLEL run of {len(self.tasks)} task(s).")
        outputs: List[Optional[str]] = [None] * len(self.tasks)

        def _run_task(index: int, task: Task) -> None:
            self._log(f"[Thread] Starting task {index + 1}: {task.description[:60]}...")
            outputs[index] = task.run()
            self._log(f"[Thread] Task {index + 1} done.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(_run_task, i, task)
                for i, task in enumerate(self.tasks)
            ]
            concurrent.futures.wait(futures)
            # Re-raise any exceptions from threads
            for f in futures:
                f.result()

        self._log("All tasks finished.")
        return outputs  # type: ignore[return-value]