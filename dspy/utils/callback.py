import functools
import inspect
import logging
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Callable

import dspy
import dspy.utils.callback_context as callback_context

ACTIVE_CALL_ID = callback_context.ACTIVE_CALL_ID

logger = logging.getLogger(__name__)

_INTERPRETER_OPERATIONS = {
    "execute": "execute",
    "invoke_tool": "tool_call",
    "start": "startup",
    "shutdown": "shutdown",
}


class BaseCallback:
    """A base class for defining callback handlers for DSPy components.

    To use a callback, subclass this class and implement the desired handlers. Each handler
    will be called at the appropriate time before/after the execution of the corresponding component.  For example, if
    you want to print a message before and after an LM is called, implement `the on_llm_start` and `on_lm_end` handler.
    Users can set the callback globally using `dspy.configure` or locally by passing it to the component
    constructor.


    Example 1: Set a global callback using `dspy.configure`.

    ```
    import dspy
    from dspy.utils.callback import BaseCallback

    class LoggingCallback(BaseCallback):

        def on_lm_start(self, call_id, instance, inputs):
            print(f"LM is called with inputs: {inputs}")

        def on_lm_end(self, call_id, outputs, exception):
            print(f"LM is finished with outputs: {outputs}")

    dspy.configure(
        callbacks=[LoggingCallback()]
    )

    cot = dspy.ChainOfThought("question -> answer")
    cot(question="What is the meaning of life?")

    # > LM is called with inputs: {'question': 'What is the meaning of life?'}
    # > LM is finished with outputs: {'answer': '42'}
    ```

    Example 2: Set a local callback by passing it to the component constructor.

    ```
    lm_1 = dspy.LM("gpt-3.5-turbo", callbacks=[LoggingCallback()])
    lm_1(question="What is the meaning of life?")

    # > LM is called with inputs: {'question': 'What is the meaning of life?'}
    # > LM is finished with outputs: {'answer': '42'}

    lm_2 = dspy.LM("gpt-3.5-turbo")
    lm_2(question="What is the meaning of life?")
    # No logging here because only `lm_1` has the callback set.
    ```
    """

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when forward() method of a module (subclass of dspy.Module) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Module instance.
            inputs: The inputs to the module's forward() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_module_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: BaseException | None = None,
    ):
        """A handler triggered after forward() method of a module (subclass of dspy.Module) is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the module's forward() method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when __call__ method of dspy.LM instance is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The LM instance.
            inputs: The inputs to the LM's __call__ method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: BaseException | None = None,
    ):
        """A handler triggered after __call__ method of dspy.LM instance is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the LM's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_adapter_format_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when format() method of an adapter (subclass of dspy.Adapter) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Adapter instance.
            inputs: The inputs to the Adapter's format() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_adapter_format_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: BaseException | None = None,
    ):
        """A handler triggered after format() method of an adapter (subclass of dspy.Adapter) is called..

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Adapter's format() method. If the method is interrupted
                by an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_adapter_parse_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when parse() method of an adapter (subclass of dspy.Adapter) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Adapter instance.
            inputs: The inputs to the Adapter's parse() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_adapter_parse_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: BaseException | None = None,
    ):
        """A handler triggered after parse() method of an adapter (subclass of dspy.Adapter) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Adapter's parse() method. If the method is interrupted
                by an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when a tool is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Tool instance.
            inputs: The inputs to the Tool's __call__ method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_tool_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: BaseException | None = None,
    ):
        """A handler triggered after a tool is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Tool's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_evaluate_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when evaluation is started.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Evaluate instance.
            inputs: The inputs to the Evaluate's __call__ method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_evaluate_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: BaseException | None = None,
    ):
        """A handler triggered after evaluation is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Evaluate's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_interpreter_execute_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        pass

    def on_interpreter_execute_end(self, call_id: str, outputs: Any | None, exception: BaseException | None = None):
        pass

    def on_interpreter_tool_call_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        pass

    def on_interpreter_tool_call_end(self, call_id: str, outputs: Any | None, exception: BaseException | None = None):
        pass

    def on_interpreter_startup_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        pass

    def on_interpreter_startup_end(self, call_id: str, outputs: Any | None, exception: BaseException | None = None):
        pass

    def on_interpreter_shutdown_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        pass

    def on_interpreter_shutdown_end(self, call_id: str, outputs: Any | None, exception: BaseException | None = None):
        pass

    def on_compile_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        """A handler triggered when an optimizer's compile method is called.

        Args:
            call_id: A unique identifier for the compile call. Can be used to connect start/end handlers.
            instance: The optimizer instance.
            inputs: The inputs to the optimizer's compile method as key-value pairs.
        """
        pass

    def on_compile_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None):
        """A handler triggered after an optimizer's compile method is called.

        Args:
            call_id: A unique identifier for the compile call. Can be used to connect start/end handlers.
            outputs: The compiled program, or None if compilation failed.
            exception: The exception raised during compilation, if any.
        """
        pass

    def on_start(self, call):
        """Observe one live invocation. Treat it as read-only; inputs may not yet be prepared.

        The default dispatches to the existing kind-specific callback API.
        """
        kind = "module" if call.kind == "predict" else call.kind
        getattr(self, f"on_{kind}_start")(call_id=call.call_id, instance=call.instance, inputs=call.arguments)

    def on_end(self, call):
        """Observe the completed invocation, including parent, timing, inputs and outcome.

        No start-event lookup is needed. The default preserves legacy callbacks.
        """
        kind = "module" if call.kind == "predict" else call.kind
        getattr(self, f"on_{kind}_end")(call_id=call.call_id, outputs=call.outputs, exception=call.exception)


def with_callbacks(fn):
    """Observe a call once, projecting it into callbacks and requested collectors."""

    @contextmanager
    def observe(instance, args, kwargs):
        from dspy.utils.execution import Invocation, _Capture

        active = callback_context.ACTIVE_INVOCATION.get()
        is_predict = isinstance(instance, dspy.Predict)
        # Predict.forward is also callable directly. Inside Module.__call__ it
        # contributes prepared inputs to that invocation, rather than recording twice.
        if is_predict and fn.__name__ in ("forward", "aforward") and active is not None and active.instance is instance:
            yield None
            return

        callbacks = dspy.settings.get("callbacks", []) + getattr(instance, "callbacks", [])
        capture = dspy.settings.get("_execution_capture")
        collect_prediction = isinstance(instance, dspy.Module) and dspy.settings.trace is not None and dspy.settings.max_trace_size > 0
        account_usage = dspy.settings.usage_tracker is not None or dspy.settings.track_usage
        if not callbacks and not (capture and (capture.recording or capture.retain)) and not collect_prediction and not account_usage:
            yield None
            return

        # Ordinary trace collection needs component identity, not serialized spans.
        # The root scope is inherited by nested calls and DSPy workers.
        with nullcontext(capture) if capture is not None else _Capture(recording=False, retain=False) as capture:
            inputs = inspect.getcallargs(fn, instance, *args, **kwargs)
            if "self" in inputs:
                inputs.pop("self")
            elif "instance" in inputs:
                inputs.pop("instance")
            kind = _call_kind(instance, fn)
            invocation = Invocation(instance, "predict" if is_predict else kind, inputs, callback_context.ACTIVE_CALL_ID.get())
            invocation.children = () if capture.retain else None
            invocation.usage = {} if capture.retain or capture.recording or account_usage else None
            if active is not None and active.children is not None:
                with active._lock:
                    active.children += (invocation,)
            capture.start(invocation)
            for callback in callbacks:
                try:
                    # Some SDK callbacks return no-ops from __getattr__ for unknown
                    # hooks. Only a declared typed hook opts out of legacy dispatch.
                    getattr(type(callback), "on_start", BaseCallback.on_start)(callback, invocation)
                except Exception as e:
                    logger.warning(f"Error when calling callback {callback}: {e}")

            # Observer code runs in the parent's context, not as a child of the call it observes.
            id_token = callback_context.ACTIVE_CALL_ID.set(invocation.call_id)
            invocation_token = callback_context.ACTIVE_INVOCATION.set(invocation)
            try:
                yield invocation
            except BaseException as e:
                invocation.exception = e
                raise
            finally:
                callback_context.ACTIVE_INVOCATION.reset(invocation_token)
                callback_context.ACTIVE_CALL_ID.reset(id_token)
                invocation.end_time = time.time()
                if is_predict:
                    instance._record_trace(invocation)
                if active is not None and active.end_time is None:
                    active._add_usage(invocation.usage or {})
                # Compatibility accounting scopes consume the same completed LM
                # observation as callbacks and retained execution trees.
                tracker = dspy.settings.usage_tracker
                accounting_error = None
                if invocation.kind == "lm" and tracker is not None:
                    try:
                        for model, usage in invocation.own_usage.items():
                            tracker.add_usage(model, usage)
                    except Exception as error:
                        if invocation.exception is None:
                            invocation.exception = accounting_error = error
                        else:
                            invocation.exception.usage_errors = (*getattr(invocation.exception, "usage_errors", ()), error)
                capture.end(invocation)
                for callback in callbacks:
                    try:
                        getattr(type(callback), "on_end", BaseCallback.on_end)(callback, invocation)
                    except Exception as e:
                        logger.warning(f"Error when applying callback {callback}'s end handler on function {fn.__name__}: {e}.")
                if accounting_error is not None:
                    raise accounting_error

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(instance, *args, **kwargs):
            with observe(instance, args, kwargs) as invocation:
                results = await fn(instance, *args, **kwargs)
                if invocation is not None:
                    invocation.outputs = results
                return results

        return async_wrapper

    else:

        @functools.wraps(fn)
        def sync_wrapper(instance, *args, **kwargs):
            with observe(instance, args, kwargs) as invocation:
                results = fn(instance, *args, **kwargs)
                if invocation is not None:
                    invocation.outputs = results
                return results

        return sync_wrapper


def _call_kind(instance: Any, fn: Callable) -> str:
    """Classify the boundary once for typed records and legacy callback dispatch."""
    from dspy.teleprompt.teleprompt import Teleprompter

    if isinstance(instance, dspy.BaseLM) or getattr(instance, "supports_decision_requests", False) is True:
        return "lm"
    elif isinstance(instance, dspy.Evaluate):
        return "evaluate"
    elif isinstance(instance, dspy.CodeInterpreter):
        return f"interpreter_{_INTERPRETER_OPERATIONS[fn.__name__]}"
    elif isinstance(instance, Teleprompter):
        return "compile"

    if isinstance(instance, dspy.Adapter):
        if fn.__name__ == "format":
            return "adapter_format"
        elif fn.__name__ == "parse":
            return "adapter_parse"
        else:
            raise ValueError(f"Unsupported adapter method for using callback: {fn.__name__}.")

    if isinstance(instance, dspy.Tool):
        return "tool"

    # We treat everything else as a module.
    return "module"
