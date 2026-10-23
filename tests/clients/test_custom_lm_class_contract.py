"""The custom-LM-class trust gate is entry-point independent.

`BaseLM.load_state` records the concrete LM class in `_dspy_lm_class` and
refuses to import a custom (non-builtin) class unless `allow_custom_lm_class`
is set, because importing it runs application code from the serialized path.
Historically the entire gate lived inside an `if cls is BaseLM:` branch, so
calling `load_state` through a subclass entry point — most notably
`dspy.LM.load_state`, which forwards via `super()` — skipped the gate and
silently constructed a plain instance of the subclass. These tests pin the
fix: the gate fires regardless of the entry point.
"""

import json

import pytest

import dspy
from dspy.clients.base_lm import LM_CLASS_STATE_KEY


class CustomBaseLM(dspy.BaseLM):
    """A custom BaseLM subclass with extra persistent state."""

    def __init__(self, model, *, deployment, **kwargs):
        super().__init__(model=model, **kwargs)
        self.deployment = deployment

    def dump_state(self):
        state = super().dump_state()
        state["deployment"] = self.deployment
        return state

    @classmethod
    def load_state(cls, state):
        state = dict(state)
        state.pop(LM_CLASS_STATE_KEY, None)
        return cls(**state)


class CustomLMSubclass(dspy.LM):
    """A custom dspy.LM subclass that adds no extra state of its own."""

    pass


class StrictLoaderBaseLM(dspy.BaseLM):
    """A custom BaseLM whose load_state does NOT strip the internal marker.

    Its constructor rejects the unexpected ``_dspy_lm_class`` kwarg, so a
    loader that forwards the saved state verbatim (``cls(**state)``) only
    works when the entry point removes the marker before dispatch.
    """

    def __init__(self, model, *, deployment, **kwargs):
        if LM_CLASS_STATE_KEY in kwargs:
            raise TypeError(f"unexpected keyword argument '{LM_CLASS_STATE_KEY}'")
        super().__init__(model=model, **kwargs)
        self.deployment = deployment

    def dump_state(self):
        state = super().dump_state()
        state["deployment"] = self.deployment
        return state

    @classmethod
    def load_state(cls, state):
        # Trusts the entry point to strip internal fields; does not pop the
        # marker itself, mirroring loaders that simply do cls(**state).
        return cls(**state)


class Echo:
    """A stateful custom engine for the custom-engine round-trip tests."""

    def __init__(self, tag="t"):
        self.tag = tag

    def complete(self, request):
        from dspy.lm15 import Message, Response, Usage

        return Response(
            id=None,
            model=request.model,
            message=Message.assistant(f"echo:{self.tag}"),
            finish_reason="stop",
            usage=Usage(),
        )

    def dump_state(self):
        return {"tag": self.tag}

    @classmethod
    def load_state(cls, state):
        return cls(**state)


class Outer:
    class NestedCustomLM(dspy.BaseLM):
        pass


class NotAnLM:
    """A module-level class that is not a BaseLM subclass."""


# ─── the gate is entry-point independent ──────────────────────────────


def test_lm_load_state_refuses_custom_lm_class_without_opt_in():
    state = CustomBaseLM(model="custom-model", deployment="prod").dump_state()
    assert state[LM_CLASS_STATE_KEY] == f"{__name__}.CustomBaseLM"

    with pytest.raises(ValueError, match="Refusing to import custom serialized LM class"):
        dspy.LM.load_state(state)


def test_baselm_and_lm_entry_points_gate_the_custom_lm_class_identically():
    state = CustomBaseLM(model="custom-model", deployment="prod").dump_state()
    # Both entry points must refuse a custom LM class by default, mirroring
    # test_loading_a_custom_engine_is_gated_like_a_custom_lm_class.
    with pytest.raises(ValueError, match="Refusing to import custom serialized LM class"):
        dspy.BaseLM.load_state(state)
    with pytest.raises(ValueError, match="Refusing to import custom serialized LM class"):
        dspy.LM.load_state(state)


def test_lm_load_state_reconstructs_custom_lm_class_with_opt_in():
    state = CustomBaseLM(model="custom-model", deployment="prod").dump_state()

    loaded = dspy.LM.load_state(state, allow_custom_lm_class=True)

    assert isinstance(loaded, CustomBaseLM)
    assert loaded.model == "custom-model"
    assert loaded.deployment == "prod"
    assert loaded.dump_state()[LM_CLASS_STATE_KEY] == f"{__name__}.CustomBaseLM"


def test_baselm_load_state_reconstructs_custom_lm_class_with_opt_in():
    state = CustomBaseLM(model="custom-model", deployment="prod").dump_state()

    loaded = dspy.BaseLM.load_state(state, allow_custom_lm_class=True)

    assert isinstance(loaded, CustomBaseLM)
    assert loaded.deployment == "prod"


def test_custom_lm_subclass_of_lm_is_gated_through_lm_load_state():
    # A state dumped from a dspy.LM subclass carries that subclass's path; it
    # must not be silently flattened into a plain dspy.LM.
    state = CustomLMSubclass(model="openai/gpt-4o-mini", engine="litellm").dump_state()
    assert state[LM_CLASS_STATE_KEY] == f"{__name__}.CustomLMSubclass"

    with pytest.raises(ValueError, match="Refusing to import custom serialized LM class"):
        dspy.LM.load_state(state)

    loaded = dspy.LM.load_state(state, allow_custom_lm_class=True)
    assert isinstance(loaded, CustomLMSubclass)
    assert loaded.engine == "litellm"


def test_custom_lm_subclass_with_custom_engine_round_trips_through_lm_load_state():
    # A custom dspy.LM subclass carrying a custom engine: the outer LM.load_state
    # must dispatch to the recorded subclass before _load_engine turns the
    # engine record into an object, otherwise the inherited LM.load_state re-entry
    # raises (the record is no longer a dict). Regression for the trust gate
    # through the subclass entry point with a custom engine.
    state = CustomLMSubclass("custom/echo", engine=Echo("a"), cache=False).dump_state()
    assert state[LM_CLASS_STATE_KEY] == f"{__name__}.CustomLMSubclass"
    assert state["engine"] == {"class": f"{__name__}:Echo", "state": {"tag": "a"}}

    with pytest.raises(ValueError, match="Refusing to import custom serialized LM class"):
        dspy.LM.load_state(state)

    loaded = dspy.LM.load_state(state, allow_custom_lm_class=True)
    assert isinstance(loaded, CustomLMSubclass)
    assert isinstance(loaded.engine, Echo) and loaded.engine.tag == "a"
    assert loaded("hi") == ["echo:a"]


def test_lm_load_state_strips_marker_before_dispatching_to_custom_loader():
    # A custom loader that does cls(**state) without popping the internal
    # marker must still load through the LM subclass entry point: the marker
    # is removed before the early dispatch, so the recorded class's
    # load_state/constructor never sees _dspy_lm_class. Regression for the
    # divergence between LM.load_state (get) and BaseLM.load_state (pop).
    state = StrictLoaderBaseLM(model="custom-model", deployment="prod").dump_state()
    assert state[LM_CLASS_STATE_KEY] == f"{__name__}.StrictLoaderBaseLM"

    loaded = dspy.LM.load_state(state, allow_custom_lm_class=True)
    assert isinstance(loaded, StrictLoaderBaseLM)
    assert loaded.deployment == "prod"
    assert LM_CLASS_STATE_KEY not in loaded.kwargs

    # The BaseLM entry point must keep behaving identically.
    loaded_baselm = dspy.BaseLM.load_state(state, allow_custom_lm_class=True)
    assert isinstance(loaded_baselm, StrictLoaderBaseLM)
    assert loaded_baselm.deployment == "prod"


def test_nested_custom_lm_class_round_trips_through_lm_load_state():
    state = Outer.NestedCustomLM(model="nested-model").dump_state()
    assert state[LM_CLASS_STATE_KEY] == f"{__name__}.Outer.NestedCustomLM"

    with pytest.raises(ValueError, match="Refusing to import custom serialized LM class"):
        dspy.LM.load_state(state)

    loaded = dspy.LM.load_state(state, allow_custom_lm_class=True)
    assert isinstance(loaded, Outer.NestedCustomLM)


# ─── the built-in path is unchanged ────────────────────────────────────


def test_builtin_marker_through_lm_load_state_is_not_gated():
    # A built-in dspy.LM marker loading through dspy.LM.load_state must still
    # construct a plain dspy.LM without requiring the trust flag.
    state = dspy.LM("openai/gpt-4o-mini", engine="litellm").dump_state()
    assert state[LM_CLASS_STATE_KEY] == "dspy.clients.lm.LM"

    loaded = dspy.LM.load_state(state)
    assert isinstance(loaded, dspy.LM)
    assert loaded.engine == "litellm"
    assert loaded.dump_state() == state


def test_builtin_marker_through_baselm_load_state_is_not_gated():
    state = dspy.LM("openai/gpt-4o-mini", engine="litellm").dump_state()

    loaded = dspy.BaseLM.load_state(state)
    assert isinstance(loaded, dspy.LM)
    assert loaded.engine == "litellm"


def test_legacy_markerless_state_through_lm_load_state_loads_as_lm():
    # A marker-less (legacy) state loading through the subclass entry point
    # still constructs cls directly, exactly as before.
    state = dspy.LM("openai/gpt-4o-mini", engine="litellm").dump_state()
    state.pop(LM_CLASS_STATE_KEY)

    loaded = dspy.LM.load_state(state)
    assert isinstance(loaded, dspy.LM)
    assert loaded.model == "openai/gpt-4o-mini"


# ─── validation of the recorded class ──────────────────────────────────


def test_recorded_class_that_is_not_a_baselm_subclass_is_typeerror():
    state = CustomBaseLM(model="custom-model", deployment="prod").dump_state()
    state[LM_CLASS_STATE_KEY] = f"{__name__}.NotAnLM"

    with pytest.raises(TypeError, match=r"must be a subclass of dspy\.BaseLM"):
        dspy.LM.load_state(state, allow_custom_lm_class=True)
    with pytest.raises(TypeError, match=r"must be a subclass of dspy\.BaseLM"):
        dspy.BaseLM.load_state(state, allow_custom_lm_class=True)


def test_unimportable_recorded_class_is_importerror_through_lm_load_state():
    state = CustomBaseLM(model="custom-model", deployment="prod").dump_state()
    state[LM_CLASS_STATE_KEY] = "no.such.module.CustomLM"

    with pytest.raises(ImportError, match="Could not import serialized LM class"):
        dspy.LM.load_state(state, allow_custom_lm_class=True)


# ─── end-to-end through dump_state / load_state round trips ────────────


def test_custom_lm_state_is_json_portable_through_lm_load_state():
    state = CustomBaseLM(model="custom-model", deployment="prod").dump_state()
    json.dumps(state)  # the gate must operate on JSON-portable state

    loaded = dspy.LM.load_state(json.loads(json.dumps(state)), allow_custom_lm_class=True)
    assert isinstance(loaded, CustomBaseLM)
    assert loaded.deployment == "prod"


def test_program_save_and_load_with_custom_lm_class_via_lm_entry(tmp_path):
    program = dspy.Predict("question -> answer")
    program.lm = CustomBaseLM(model="custom-model", deployment="saved")
    path = tmp_path / "program.json"
    program.save(path)

    fresh = dspy.Predict("question -> answer")
    with pytest.raises(ValueError, match="Refusing to import custom serialized LM class"):
        fresh.load(path)
    fresh.load(path, allow_unsafe_lm_state=True)
    assert isinstance(fresh.lm, CustomBaseLM)
    assert fresh.lm.deployment == "saved"
