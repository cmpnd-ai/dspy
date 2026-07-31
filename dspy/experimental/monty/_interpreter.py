"""Private persistent Monty CodeInterpreter adapter."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput

from ._bridge import Invocation, jsonable
from ._shim import SHIM


class MontyInterpreter:
    execution_instructions = """This is a constrained Monty Python environment, not CPython.
Use plain functions, loops, comprehensions, and built-in containers. Available modules include json, math, re,
datetime, pathlib, typing, and limited asyncio. Do not use package installation, subprocesses, threads, sockets,
eval/exec, runtime introspection, class inheritance, properties, generators, or custom magic methods. Prefer
approved host tools and llm_query_batched for I/O and concurrent semantic work. The capability-limited dspy facade
supports Predict, ChainOfThought, RLM, CodeAct, ProgramOfThought, ReAct, and ReActV2."""

    def __init__(self, invocation: Invocation):
        self.invocation = invocation
        self.tools = {}
        self.output_fields = None
        self._pool = self._checkout = self._session = None
        self._closed = False
        self._terminal_error = None

    def start(self):
        if self._session is not None:
            return
        if self._terminal_error is not None:
            raise CodeInterpreterError("MontyInterpreter session has ended; create a new interpreter for a fresh session.")
        if self._closed:
            raise CodeInterpreterError("interpreter has been shut down")
        try:
            from pydantic_monty import Monty
        except ImportError as error:
            raise ImportError("Monty support requires `pip install 'dspy[monty]'`") from error
        pool = Monty(request_timeout=self.invocation.policy.request_timeout)
        try:
            pool.__enter__()
            checkout = pool.checkout(limits=self.invocation.policy.limits)
            session = checkout.__enter__()
        except Exception as error:
            try:
                pool.__exit__(None, None, None)
            except Exception:
                pass
            raise CodeInterpreterError(f"failed to start Monty interpreter: {error}") from error
        self._pool, self._checkout, self._session = pool, checkout, session

    def execute(self, code: str, variables: dict[str, Any] | None = None):
        self.start()
        names = [field["name"] for field in (self.output_fields or [])]
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            tree.body[-1] = ast.copy_location(
                ast.Assign(targets=[ast.Name(id="__dspy_result", ctx=ast.Store())], value=tree.body[-1].value),
                tree.body[-1],
            )
            ast.fix_missing_locations(tree)
            code = ast.unparse(tree)
        wrapper = f"""\nimport json as __dspy_json
def SUBMIT(*args, **kwargs):
    names = {names!r}
    if names:
        if args:
            if kwargs or len(args) != len(names): raise ValueError("SUBMIT arguments do not match output fields")
            kwargs = dict(zip(names, args))
        elif set(kwargs) != set(names): raise ValueError("SUBMIT arguments do not match output fields")
    else:
        if kwargs or len(args) != 1: raise ValueError("SUBMIT requires exactly one positional output")
        kwargs = {{"output": args[0]}}
    raise BaseException("__DSPY_FINAL__" + __dspy_json.dumps(kwargs))
__dspy_result = None
try:
{textwrap.indent(code, "    ")}
except BaseException as error:
    __dspy_message = str(error)
    if not __dspy_message.startswith("__DSPY_FINAL__"): raise
    __dspy_result = {{"__dspy_final__": __dspy_json.loads(__dspy_message[14:])}}
__dspy_result
"""
        printed = []
        lookup = self.invocation.lookup()
        for name, tool in self.tools.items():
            if name in lookup:
                # Invocation-owned tools retain their aggregate-budget wrappers.
                continue
            if name in {"llm_query", "llm_query_batched"}:
                lookup[name] = tool
                continue

            def invoke(*args, _tool=tool, **kwargs):
                self.invocation.budget.count("tool")
                arguments = inspect.signature(_tool).bind(*args, **kwargs).arguments
                return jsonable(_tool(**arguments))

            lookup[name] = invoke
        try:
            from pydantic_monty import MontyCrashedError, MontyError

            result = self._session.feed_run(
                SHIM + wrapper,
                inputs=jsonable(variables or {}),
                external_lookup=lookup,
                print_callback=lambda _stream, value: printed.append(value),
            )
        except MontyCrashedError as error:
            message = f"{error}; interpreter state was lost. Create a new interpreter for a fresh session."
            try:
                self.shutdown()
            except Exception:
                pass
            self._terminal_error = message
            raise CodeInterpreterError(message) from error
        except MontyError as error:
            raise CodeExecutionError(str(error)) from error
        except Exception as error:
            raise CodeInterpreterError(str(error)) from error
        if isinstance(result, dict) and "__dspy_final__" in result:
            return FinalOutput(result["__dspy_final__"])
        return "".join(printed).rstrip("\n") if printed else result

    def shutdown(self):
        if self._closed:
            return
        checkout, pool = self._checkout, self._pool
        self._pool = self._checkout = self._session = None
        self._closed = True
        try:
            if checkout is not None:
                checkout.__exit__(None, None, None)
        finally:
            if pool is not None:
                pool.__exit__(None, None, None)
