import functools
import inspect
import logging
import uuid
from contextvars import ContextVar
from typing import Any, Callable

import dspy

ACTIVE_CALL_ID = ContextVar("active_call_id", default=None)

logger = logging.getLogger(__name__)


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
        exception: Exception | None = None,
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
        exception: Exception | None = None,
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
        exception: Exception | None = None,
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
        exception: Exception | None = None,
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
        exception: Exception | None = None,
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
        exception: Exception | None = None,
    ):
        """A handler triggered after evaluation is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Evaluate's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_interpreter_execute_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when execute() method of a CodeInterpreter is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The CodeInterpreter instance (e.g. dspy.PythonInterpreter).
            inputs: The inputs to the interpreter's execute() method, stored as key-value pairs.
                Includes ``code`` (the Python source to run) and ``variables`` (the namespace
                injected before execution).

                Note: ``inputs["variables"]`` can be very large. For example, dspy.RLM passes its
                input data through it on every iteration. The framework does not truncate or copy
                payloads, mirroring the ``on_lm_start`` precedent of passing full messages, so
                handlers are responsible for any truncation or redaction they need.
        """
        pass

    def on_interpreter_execute_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after execute() method of a CodeInterpreter is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the interpreter's execute() method. This may be a
                ``FinalOutput`` (when SUBMIT() was called), a string, a list, or None. If the
                method is interrupted by an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
                Interpreter execution errors surface as ``CodeInterpreterError`` and invalid
                Python source surfaces as ``SyntaxError``.
        """
        pass

    def on_interpreter_tool_call_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when the sandbox invokes a host-side tool.

        This fires for every entry in ``interpreter.tools``, including plain closures such as
        dspy.RLM's ``llm_query``/``llm_query_batched``, not just dspy.Tool objects.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The CodeInterpreter instance that dispatched the tool call.
            inputs: The inputs to the tool dispatch, stored as key-value pairs. Includes
                ``tool_name`` (the name of the invoked tool) and ``kwargs`` (the keyword
                arguments passed from the sandbox).
        """
        pass

    def on_interpreter_tool_call_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after the sandbox invokes a host-side tool.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The value returned by the tool. If the tool raises, this will be None.
            exception: If the tool raises during dispatch, the exception is stored here. The
                interpreter still converts it into a sandbox-side error afterwards, so this
                handler observes the real exception without changing sandbox behavior.
        """
        pass

    def on_interpreter_startup_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when start() method of a CodeInterpreter is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The CodeInterpreter instance being started.
            inputs: The inputs to the interpreter's start() method, stored as key-value pairs.
        """
        pass

    def on_interpreter_startup_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after start() method of a CodeInterpreter is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the interpreter's start() method. If the method is interrupted
                by an exception, this will be None.
            exception: If an exception is raised during startup (e.g. the sandbox process fails to
                spawn), it will be stored here.
        """
        pass

    def on_interpreter_shutdown_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when shutdown() method of a CodeInterpreter is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The CodeInterpreter instance being shut down.
            inputs: The inputs to the interpreter's shutdown() method, stored as key-value pairs.
        """
        pass

    def on_interpreter_shutdown_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after shutdown() method of a CodeInterpreter is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the interpreter's shutdown() method. If the method is
                interrupted by an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass


def with_callbacks(fn):
    """Decorator to add callback functionality to instance methods."""

    def _execute_start_callbacks(instance, fn, call_id, callbacks, args, kwargs):
        """Execute all start callbacks for a function call."""
        inputs = inspect.getcallargs(fn, instance, *args, **kwargs)
        if "self" in inputs:
            inputs.pop("self")
        elif "instance" in inputs:
            inputs.pop("instance")
        for callback in callbacks:
            try:
                _get_on_start_handler(callback, instance, fn)(call_id=call_id, instance=instance, inputs=inputs)
            except Exception as e:
                logger.warning(f"Error when calling callback {callback}: {e}")

    def _execute_end_callbacks(instance, fn, call_id, results, exception, callbacks):
        """Execute all end callbacks for a function call."""
        for callback in callbacks:
            try:
                _get_on_end_handler(callback, instance, fn)(
                    call_id=call_id,
                    outputs=results,
                    exception=exception,
                )
            except Exception as e:
                logger.warning(f"Error when applying callback {callback}'s end handler on function {fn.__name__}: {e}.")

    def _get_active_callbacks(instance):
        """Get combined global and instance-level callbacks."""
        return dspy.settings.get("callbacks", []) + getattr(instance, "callbacks", [])

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(instance, *args, **kwargs):
            callbacks = _get_active_callbacks(instance)
            if not callbacks:
                return await fn(instance, *args, **kwargs)

            call_id = uuid.uuid4().hex

            _execute_start_callbacks(instance, fn, call_id, callbacks, args, kwargs)

            # Active ID must be set right before the function is called, not before calling the callbacks.
            parent_call_id = ACTIVE_CALL_ID.get()
            ACTIVE_CALL_ID.set(call_id)

            results = None
            exception = None
            try:
                results = await fn(instance, *args, **kwargs)
                return results
            except Exception as e:
                exception = e
                raise exception
            finally:
                ACTIVE_CALL_ID.set(parent_call_id)
                _execute_end_callbacks(instance, fn, call_id, results, exception, callbacks)

        return async_wrapper

    else:

        @functools.wraps(fn)
        def sync_wrapper(instance, *args, **kwargs):
            callbacks = _get_active_callbacks(instance)
            if not callbacks:
                return fn(instance, *args, **kwargs)

            call_id = uuid.uuid4().hex

            _execute_start_callbacks(instance, fn, call_id, callbacks, args, kwargs)

            # Active ID must be set right before the function is called, not before calling the callbacks.
            parent_call_id = ACTIVE_CALL_ID.get()
            ACTIVE_CALL_ID.set(call_id)

            results = None
            exception = None
            try:
                results = fn(instance, *args, **kwargs)
                return results
            except Exception as e:
                exception = e
                raise exception
            finally:
                ACTIVE_CALL_ID.set(parent_call_id)
                _execute_end_callbacks(instance, fn, call_id, results, exception, callbacks)

        return sync_wrapper


# Maps the decorated CodeInterpreter method name to the corresponding callback handler. The
# tool-call hook is routed via the extracted `_invoke_tool` seam (see PythonInterpreter) rather
# than the exception-swallowing `_handle_tool_call`, so the end handler observes real exceptions.
_INTERPRETER_START_HANDLERS = {
    "execute": "on_interpreter_execute_start",
    "start": "on_interpreter_startup_start",
    "shutdown": "on_interpreter_shutdown_start",
    "_invoke_tool": "on_interpreter_tool_call_start",
}

_INTERPRETER_END_HANDLERS = {
    "execute": "on_interpreter_execute_end",
    "start": "on_interpreter_startup_end",
    "shutdown": "on_interpreter_shutdown_end",
    "_invoke_tool": "on_interpreter_tool_call_end",
}


def _get_on_start_handler(callback: BaseCallback, instance: Any, fn: Callable) -> Callable:
    """Selects the appropriate on_start handler of the callback based on the instance and function name."""
    if isinstance(instance, dspy.BaseLM):
        return callback.on_lm_start
    elif isinstance(instance, dspy.Evaluate):
        return callback.on_evaluate_start

    if isinstance(instance, dspy.Adapter):
        if fn.__name__ == "format":
            return callback.on_adapter_format_start
        elif fn.__name__ == "parse":
            return callback.on_adapter_parse_start
        else:
            raise ValueError(f"Unsupported adapter method for using callback: {fn.__name__}.")

    if isinstance(instance, dspy.Tool):
        return callback.on_tool_start

    # CodeInterpreter is a runtime_checkable Protocol, so this isinstance check is structural
    # (it only checks for the presence of start/execute/shutdown/tools). None of the branches
    # above are structural interpreters, so ordering relative to them is safe; this branch must
    # stay before the module fallback so interpreter events do not masquerade as module events.
    if isinstance(instance, dspy.CodeInterpreter):
        handler_name = _INTERPRETER_START_HANDLERS.get(fn.__name__)
        if handler_name is None:
            raise ValueError(f"Unsupported interpreter method for using callback: {fn.__name__}.")
        return getattr(callback, handler_name)

    # We treat everything else as a module.
    return callback.on_module_start


def _get_on_end_handler(callback: BaseCallback, instance: Any, fn: Callable) -> Callable:
    """Selects the appropriate on_end handler of the callback based on the instance and function name."""
    if isinstance(instance, dspy.BaseLM):
        return callback.on_lm_end
    elif isinstance(instance, dspy.Evaluate):
        return callback.on_evaluate_end

    if isinstance(instance, (dspy.Adapter)):
        if fn.__name__ == "format":
            return callback.on_adapter_format_end
        elif fn.__name__ == "parse":
            return callback.on_adapter_parse_end
        else:
            raise ValueError(f"Unsupported adapter method for using callback: {fn.__name__}.")

    if isinstance(instance, dspy.Tool):
        return callback.on_tool_end

    # See the note in _get_on_start_handler: this structural Protocol check must stay before the
    # module fallback so interpreter events do not masquerade as module events.
    if isinstance(instance, dspy.CodeInterpreter):
        handler_name = _INTERPRETER_END_HANDLERS.get(fn.__name__)
        if handler_name is None:
            raise ValueError(f"Unsupported interpreter method for using callback: {fn.__name__}.")
        return getattr(callback, handler_name)

    # We treat everything else as a module.
    return callback.on_module_end
