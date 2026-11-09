"""
Tool registry: the allowlist of everything Jarvis can do.

Every tool is a plain Python function plus a description, a parameter spec and
a risk level. Tools register themselves with the @tool decorator. The model can
only call tools that are in the registry: there is no generic shell tool.

A parameter spec maps each argument name to (type, description) or
(type, description, allowed_values). Types: string, integer, number, boolean.
Optional parameters need a default value in the function.

registry.execute() is the ONLY way a tool gets run: it validates the arguments,
asks for confirmation when the tool's risk is "confirm", enforces a timeout,
catches every error and keeps the result short.
"""

import inspect
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

RISK_LEVELS = ("safe", "confirm")   # "confirm" = ask sir y/n before running
PARAM_TYPES = ("string", "integer", "number", "boolean")
MAX_RESULT_CHARS = 400              # ~100 tokens: context is only 4096
MAX_ARG_CHARS = 300                 # longest string argument we accept from the model
TRUNCATED = " ... (truncated)"


@dataclass
class ToolResult:
    """What every tool returns: did it work, plus a short message the model can use."""

    success: bool
    message: str
    confirmed: bool | None = None   # filled in by execute(): None = no confirmation needed
    duration_s: float = 0.0         # filled in by execute()


@dataclass
class Tool:
    """One registered tool."""

    name: str
    description: str
    parameters: dict                    # name -> (type, description[, allowed values])
    risk: str                           # "safe" or "confirm"
    func: Callable[..., ToolResult]
    optional: tuple = ()                # parameter names that are not required
    timeout_s: float = 10.0             # max seconds we wait for the tool

    def schema(self) -> dict:
        """The tool definition in the format Ollama (and OpenAI-style APIs) expect."""
        properties = {}
        for pname, spec in self.parameters.items():
            prop = {"type": spec[0], "description": spec[1]}
            if len(spec) > 2:
                prop["enum"] = list(spec[2])
            properties[pname] = prop
        required = [p for p in self.parameters if p not in self.optional]
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        }}


def _coerce(value, ptype: str):
    """Return value in the right type, or raise ValueError with a short reason."""
    if ptype == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise ValueError("must be true or false")

    if ptype in ("integer", "number"):
        if isinstance(value, bool):
            raise ValueError(f"must be a {ptype}")
        if isinstance(value, str):
            try:
                value = float(value.strip())  # small models sometimes send "30"
            except ValueError:
                raise ValueError(f"must be a {ptype}") from None
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"must be a {ptype}")
        if ptype == "integer":
            if not float(value).is_integer():
                raise ValueError("must be a whole number")
            return int(value)
        return value

    # string
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("must be a non-empty text")
    if len(value) > MAX_ARG_CHARS:
        raise ValueError(f"is longer than {MAX_ARG_CHARS} characters")
    return value.strip()


def _truncate(text: str) -> str:
    text = str(text).strip()
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS - len(TRUNCATED)].rstrip() + TRUNCATED


class ToolRegistry:
    """Holds all tools. Jarvis uses the module-level `registry`; tests make their own."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    # ------------------------------------------------------------ registering
    def tool(self, description: str, parameters: dict | None = None, risk: str = "safe",
             optional: tuple = (), timeout_s: float = 10.0, name: str | None = None):
        """Decorator: register the function below it as a tool. The function is returned unchanged."""
        parameters = parameters or {}
        if risk not in RISK_LEVELS:
            raise ValueError(f"Unknown risk level '{risk}', use one of {RISK_LEVELS}")
        for pname, spec in parameters.items():
            if spec[0] not in PARAM_TYPES:
                raise ValueError(f"Parameter '{pname}' has unknown type '{spec[0]}'")
        for pname in optional:
            if pname not in parameters:
                raise ValueError(f"Optional parameter '{pname}' is not in the parameters")

        def register(func: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
            tool_name = name or func.__name__
            if tool_name in self._tools:
                raise ValueError(f"Tool '{tool_name}' is registered twice")
            signature = inspect.signature(func).parameters
            if set(signature) != set(parameters):
                raise ValueError(f"Tool '{tool_name}': function arguments {sorted(signature)} "
                                 f"do not match parameters {sorted(parameters)}")
            for pname in optional:
                if signature[pname].default is inspect.Parameter.empty:
                    raise ValueError(f"Tool '{tool_name}': optional parameter '{pname}' needs a default")
            self._tools[tool_name] = Tool(tool_name, description, parameters, risk,
                                          func, tuple(optional), timeout_s)
            return func

        return register

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict]:
        """All tool definitions, ready to pass to brain.chat_stream(messages, tools=...)."""
        return [t.schema() for t in self._tools.values()]

    # ------------------------------------------------------------ running
    def execute(self, name: str, arguments: dict | None = None,
                confirm: Callable[[Tool, dict], bool] | None = None) -> ToolResult:
        """
        Run a tool the model asked for. Never raises (except Ctrl+C): every problem
        becomes ToolResult(success=False, message=...) that the model can read.

        confirm: function(tool, args) -> True/False, called for "confirm" tools.
                 No confirm function means the answer is NO.
        """
        start = time.perf_counter()
        result = self._run(name, arguments, confirm)
        result.duration_s = round(time.perf_counter() - start, 3)
        result.message = _truncate(result.message)
        return result

    def _validate(self, tool: Tool, arguments) -> tuple[dict | None, str | None]:
        """Check the model's arguments against the spec. Returns (clean_args, error)."""
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return None, "The arguments must be an object with named values."
        arguments = {k: v for k, v in arguments.items() if v is not None}

        unknown = set(arguments) - set(tool.parameters)
        if unknown:
            return None, f"Unknown argument(s): {', '.join(sorted(unknown))}."
        missing = [p for p in tool.parameters if p not in tool.optional and p not in arguments]
        if missing:
            return None, f"Missing argument(s): {', '.join(missing)}."

        clean = {}
        for pname, value in arguments.items():
            spec = tool.parameters[pname]
            try:
                value = _coerce(value, spec[0])
            except ValueError as e:
                return None, f"Argument '{pname}' {e}."
            if len(spec) > 2 and value not in spec[2]:
                return None, f"Argument '{pname}' must be one of: {', '.join(map(str, spec[2]))}."
            clean[pname] = value
        return clean, None

    def _run(self, name, arguments, confirm) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(False, f"There is no tool called '{name}'.")

        args, error = self._validate(tool, arguments)
        if error:
            return ToolResult(False, error)

        confirmed = None
        if tool.risk == "confirm":
            try:
                confirmed = bool(confirm and confirm(tool, args))
            except Exception:
                confirmed = False
            if not confirmed:
                return ToolResult(False, "Sir declined, so nothing was done.", confirmed=False)

        # Python cannot kill a stuck thread: after the timeout we stop WAITING, the thread
        # may finish in the background. Tools themselves use subprocess timeouts (step 4).
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(tool.func, **args)
        try:
            result = future.result(timeout=tool.timeout_s)
        except TimeoutError:
            return ToolResult(False, f"The tool did not finish within {tool.timeout_s:g} seconds.",
                              confirmed=confirmed)
        except Exception as e:
            return ToolResult(False, f"The tool failed: {e}", confirmed=confirmed)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        if not isinstance(result, ToolResult):
            return ToolResult(False, "The tool returned something unexpected.", confirmed=confirmed)
        result.confirmed = confirmed
        return result


# The registry Jarvis uses. Tool modules do:  from jarvis.tools.registry import tool, ToolResult
registry = ToolRegistry()
tool = registry.tool
