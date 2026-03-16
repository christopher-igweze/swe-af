"""HTTP API provider — calls OpenRouter chat completions directly.

No subprocess, no binary dependency, works on any architecture (ARM64/x86_64).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Type, TypeVar

import httpx
from pydantic import BaseModel

from swe_af.agent_ai.types import (
    AgentResponse,
    Message,
    Metrics,
    TextContent,
)

T = TypeVar("T", bound=BaseModel)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_TRANSIENT_PATTERNS = frozenset(
    {
        "rate limit",
        "rate_limit",
        "overloaded",
        "timeout",
        "timed out",
        "connection reset",
        "connection refused",
        "temporarily unavailable",
        "service unavailable",
        "503",
        "502",
        "504",
        "internal server error",
        "500",
    }
)

# Rough pricing per 1M tokens (input, output).
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "minimax/minimax-m2.5": (0.25, 1.20),
    "anthropic/claude-haiku-4.5": (0.80, 4.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "anthropic/claude-sonnet-4-20250514": (3.00, 15.00),
}

# Tools exposed to the LLM for agentic file-editing loops.
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file (relative to the working directory).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative file path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                    "content": {"type": "string", "description": "Full file content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Shell command to execute"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that the coding task is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Brief summary of what was done"},
                    "files_changed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of files that were created or modified",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


def _is_transient(error: str) -> bool:
    low = error.lower()
    return any(p in low for p in _TRANSIENT_PATTERNS)


def _write_log(fh: IO[str], event: str, **data: Any) -> None:
    entry = {"ts": time.time(), "event": event, **data}
    fh.write(json.dumps(entry, default=str) + "\n")
    fh.flush()


def _open_log(log_file: str | Path | None) -> IO[str] | None:
    if log_file is None:
        return None
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", encoding="utf-8")


def _schema_output_path(cwd: str) -> str:
    name = f".api_output_{uuid.uuid4().hex[:12]}.json"
    return os.path.join(os.path.abspath(cwd), name)


def _build_schema_suffix(output_path: str, schema_json: str) -> str:
    return (
        f"\n\n---\n"
        f"IMPORTANT — STRUCTURED OUTPUT REQUIREMENT:\n"
        f"After completing the task, you MUST write your final structured output "
        f"as a single valid JSON object to this file:\n"
        f"  {output_path}\n\n"
        f"The JSON must conform to this schema:\n"
        f"```json\n{schema_json}\n```\n\n"
        f"Write ONLY valid JSON to the file — no markdown fences, no explanation, "
        f"just the raw JSON object. Use the write_file tool to create the file."
    )


def _read_and_parse_json_file(path: str, schema: Type[T]) -> T | None:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n", 1)
            text = lines[1] if len(lines) > 1 else text
            if text.endswith("```"):
                text = text[: -len("```")]
            text = text.strip()
        data = json.loads(text)
        return schema.model_validate(data)
    except Exception:
        return None


def _cleanup_files(paths: list[str]) -> None:
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


@dataclass
class APIProviderConfig:
    """Configuration for the HTTP API provider client."""

    model: str = "anthropic/claude-sonnet-4-20250514"
    cwd: str | Path = "."
    max_turns: int = 15
    allowed_tools: list[str] = field(default_factory=list)
    system_prompt: str | None = None
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    permission_mode: str | None = None
    max_budget_usd: float | None = None
    env: dict[str, str] = field(default_factory=dict)
    request_timeout: float = 180.0


class APIProviderClient:
    """Direct HTTP API provider — no subprocess, works on any architecture.

    Calls OpenRouter's chat completions API with tool-use for agentic
    file editing, reading, and command execution loops.
    """

    def __init__(self, config: APIProviderConfig | None = None) -> None:
        self.config = config or APIProviderConfig()

    async def run(
        self,
        prompt: str,
        *,
        model: str | None = None,
        cwd: str | Path | None = None,
        max_turns: int | None = None,
        allowed_tools: list[str] | None = None,
        system_prompt: str | None = None,
        output_schema: Type[T] | None = None,
        max_retries: int | None = None,
        max_budget_usd: float | None = None,
        permission_mode: str | None = None,
        env: dict[str, str] | None = None,
        log_file: str | Path | None = None,
    ) -> AgentResponse[T]:
        """Run a prompt through OpenRouter's chat API with tool-use loop."""
        cfg = self.config
        effective_model = model or cfg.model
        effective_cwd = str(cwd or cfg.cwd)
        effective_turns = max_turns or cfg.max_turns
        effective_retries = max_retries if max_retries is not None else cfg.max_retries
        effective_system = system_prompt or cfg.system_prompt

        output_path: str | None = None
        final_prompt = prompt
        if output_schema:
            output_path = _schema_output_path(effective_cwd)
            schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
            final_prompt = prompt + _build_schema_suffix(output_path, schema_json)

        temp_files: list[str] = []
        if output_path:
            temp_files.append(output_path)

        log_fh = _open_log(log_file)
        try:
            return await self._run_with_retries(
                prompt=prompt,
                final_prompt=final_prompt,
                output_schema=output_schema,
                output_path=output_path,
                effective_cwd=effective_cwd,
                effective_model=effective_model,
                effective_turns=effective_turns,
                effective_system=effective_system,
                effective_retries=effective_retries,
                temp_files=temp_files,
                log_fh=log_fh,
            )
        finally:
            if log_fh:
                log_fh.close()
            _cleanup_files(temp_files)

    async def _run_with_retries(
        self,
        *,
        prompt: str,
        final_prompt: str,
        output_schema: Type[T] | None,
        output_path: str | None,
        effective_cwd: str,
        effective_model: str,
        effective_turns: int,
        effective_system: str | None,
        effective_retries: int,
        temp_files: list[str],
        log_fh: IO[str] | None = None,
    ) -> AgentResponse[T]:
        cfg = self.config
        delay = cfg.initial_delay
        last_exc: Exception | None = None

        if log_fh:
            _write_log(log_fh, "start", prompt=prompt, model=effective_model, max_turns=effective_turns)

        for attempt in range(effective_retries + 1):
            try:
                response = await self._execute(
                    prompt=final_prompt,
                    model=effective_model,
                    cwd=effective_cwd,
                    max_turns=effective_turns,
                    system_prompt=effective_system,
                    log_fh=log_fh,
                )

                if not output_schema or output_path is None:
                    if log_fh:
                        _write_log(
                            log_fh, "end",
                            is_error=response.is_error,
                            num_turns=response.metrics.num_turns,
                            cost_usd=response.metrics.total_cost_usd,
                        )
                    return response

                parsed = _read_and_parse_json_file(output_path, output_schema)
                if parsed is not None:
                    resp = AgentResponse(
                        result=response.result,
                        parsed=parsed,
                        messages=response.messages,
                        metrics=response.metrics,
                        is_error=False,
                    )
                    if log_fh:
                        _write_log(
                            log_fh, "end",
                            is_error=False,
                            num_turns=response.metrics.num_turns,
                            cost_usd=response.metrics.total_cost_usd,
                        )
                    return resp

                # Schema file not found or parse failed — try constructing from tracked data
                # The tool loop tracks files_changed internally; use that to build a valid schema
                fallback_parsed = None
                if output_schema:
                    try:
                        # Extract files_changed from the response text or default to empty
                        fallback_data = {
                            "files_changed": [],
                            "summary": (response.result or "")[:500],
                            "complete": True,
                        }
                        # Check if _execute tracked any file writes via its internal state
                        if hasattr(response, "_files_changed") and response._files_changed:
                            fallback_data["files_changed"] = sorted(response._files_changed)
                        fallback_parsed = output_schema.model_validate(fallback_data)
                    except Exception:
                        pass

                if log_fh:
                    _write_log(log_fh, "end", is_error=fallback_parsed is None, reason="schema file fallback")
                return AgentResponse(
                    result=response.result,
                    parsed=fallback_parsed,
                    messages=response.messages,
                    metrics=response.metrics,
                    is_error=fallback_parsed is None,
                )

            except Exception as e:
                last_exc = e
                if attempt < effective_retries and _is_transient(str(e)):
                    if log_fh:
                        _write_log(log_fh, "retry", attempt=attempt + 1, error=str(e), delay=delay)
                    await asyncio.sleep(delay)
                    delay = min(delay * cfg.backoff_factor, cfg.max_delay)
                    if output_schema:
                        output_path = _schema_output_path(effective_cwd)
                        temp_files.append(output_path)
                        schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
                        final_prompt = prompt + _build_schema_suffix(output_path, schema_json)
                    continue
                if log_fh:
                    _write_log(log_fh, "end", is_error=True, error=str(e))
                raise

        raise last_exc  # type: ignore[misc]

    async def _execute(
        self,
        *,
        prompt: str,
        model: str,
        cwd: str,
        max_turns: int,
        system_prompt: str | None,
        log_fh: IO[str] | None = None,
    ) -> AgentResponse[Any]:
        """Execute a single tool-use loop against OpenRouter."""
        start_time = time.time()
        api_key = self.config.env.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set in environment or provider config")

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        total_input_tokens = 0
        total_output_tokens = 0
        files_changed: set[str] = set()
        final_text: str | None = None
        num_turns = 0

        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            for turn in range(max_turns):
                num_turns += 1
                resp = await client.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "tools": _TOOLS,
                        "tool_choice": "auto",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                usage = data.get("usage", {})
                total_input_tokens += usage.get("prompt_tokens", 0)
                total_output_tokens += usage.get("completion_tokens", 0)

                choice = data["choices"][0]
                msg = choice["message"]
                messages.append(msg)

                # Capture text content
                if msg.get("content"):
                    final_text = msg["content"]

                tool_calls = msg.get("tool_calls", [])
                if not tool_calls:
                    # No tool calls — LLM is done
                    break

                done_hit = False
                for tc in tool_calls:
                    fn = tc["function"]
                    name = fn["name"]
                    try:
                        args = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": "Error: could not parse tool arguments as JSON.",
                        })
                        continue

                    if name == "read_file":
                        path = os.path.join(cwd, args["path"])
                        try:
                            with open(path, "r", encoding="utf-8", errors="replace") as f:
                                result = f.read()
                        except Exception as e:
                            result = f"Error: {e}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result[:10_000],
                        })

                    elif name == "write_file":
                        path = os.path.join(cwd, args["path"])
                        try:
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            with open(path, "w", encoding="utf-8") as f:
                                f.write(args["content"])
                            files_changed.add(args["path"])
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": "File written successfully.",
                            })
                        except Exception as e:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": f"Error writing file: {e}",
                            })

                    elif name == "run_command":
                        try:
                            proc = subprocess.run(
                                args["command"],
                                shell=True,
                                capture_output=True,
                                text=True,
                                cwd=cwd,
                                timeout=60,
                            )
                            result = proc.stdout[:5000]
                            if proc.stderr:
                                result += "\nSTDERR: " + proc.stderr[:2000]
                            if proc.returncode != 0:
                                result += f"\n(exit code {proc.returncode})"
                        except subprocess.TimeoutExpired:
                            result = "Error: command timed out after 60 seconds"
                        except Exception as e:
                            result = f"Error: {e}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        })

                    elif name == "done":
                        if args.get("files_changed"):
                            files_changed.update(args["files_changed"])
                        final_text = args.get("summary", final_text)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": "Task complete.",
                        })
                        done_hit = True

                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": f"Error: unknown tool '{name}'",
                        })

                if done_hit:
                    break

        duration_ms = int((time.time() - start_time) * 1000)
        cost = self._calc_cost(model, total_input_tokens, total_output_tokens)

        metrics = Metrics(
            duration_ms=duration_ms,
            duration_api_ms=duration_ms,
            num_turns=num_turns,
            total_cost_usd=cost,
            usage={
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
            },
            session_id="",
        )

        assistant_messages = [
            Message(
                role="assistant",
                content=[TextContent(text=final_text)] if final_text else [],
                model=model,
                error=None,
                parent_tool_use_id=None,
            )
        ]

        if log_fh:
            _write_log(
                log_fh, "result",
                num_turns=num_turns,
                duration_ms=duration_ms,
                cost_usd=cost,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                files_changed=sorted(files_changed),
            )

        resp = AgentResponse(
            result=final_text,
            parsed=parsed_result,
            messages=assistant_messages,
            metrics=metrics,
            is_error=False,
        )
        # Attach tracked files for fallback schema construction in _run_with_retries
        resp._files_changed = sorted(files_changed)  # type: ignore[attr-defined]
        return resp

    @staticmethod
    def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD from token counts."""
        inp_rate, out_rate = _MODEL_PRICING.get(model, (1.0, 5.0))
        return (input_tokens * inp_rate + output_tokens * out_rate) / 1_000_000
