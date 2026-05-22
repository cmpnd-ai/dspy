from dspy.adapters.types.history import (
    History,
    HistoryFrame,
    Observation,
    make_truncate_oldest_actions,
    truncate_oldest_actions,
)


def test_legacy_messages_key_still_constructs_history_frames():
    legacy_message = {"question": "What is the capital of France?", "answer": "Paris"}

    history = History(messages=[legacy_message])

    assert history.frames[0] == legacy_message
    assert history.messages is history.frames
    assert history.model_dump() == {"frames": [legacy_message]}


def test_field_frames_round_trip():
    history = History(frames=[])

    history.append_inputs({"question": "hi"})
    history.append_outputs(
        {"next_thought": "search first"},
        observations=[Observation(value="result", source="tool", call_id="call_0", name="search")],
    )
    history.append_output({"answer": "bye"})

    assert isinstance(history.frames[0], HistoryFrame)
    assert history.frames[0].inputs == {"question": "hi"}
    assert history.frames[1].outputs == {"next_thought": "search first"}
    assert history.frames[1].observations[0].call_id == "call_0"
    assert history.frames[2].outputs == {"answer": "bye"}
    assert history.frames[2].complete
    assert History.model_validate(history.model_dump()) == history


def test_has_open_episode_tracks_input_and_complete_boundaries():
    history = History(messages=[{"question": "legacy", "answer": "legacy answer"}])

    assert not history.has_open_episode()

    history.append_inputs({"question": "new"})
    assert history.has_open_episode()

    history.append_outputs({"next_thought": "working"})
    history.frames.append({"question": "another legacy message"})
    assert history.has_open_episode()

    history.append_output({"answer": "done"})
    history.frames.append({"question": "final legacy message"})
    assert not history.has_open_episode()


def test_compact_if_needed_calls_compact_fn_with_history():
    calls = []
    history = History(frames=[], compact_fn=calls.append)

    history.compact_if_needed()

    assert calls == [history]


def test_truncate_oldest_actions_keeps_recent_observed_frames_and_non_actions():
    legacy_message = {"question": "legacy", "answer": "legacy answer"}
    history = History(frames=[HistoryFrame(inputs={"question": "new"}), legacy_message])

    for index in range(5):
        history.append_outputs(
            {"next_thought": str(index)},
            observations=[Observation(value=f"result {index}", call_id=f"call_{index}")],
        )
    history.append_output({"answer": "done"})

    truncate_oldest_actions(history, max_tokens=0, keep_n=2)

    observed_frames = [frame for frame in history.frames if isinstance(frame, HistoryFrame) and frame.observations]
    assert [frame.outputs["next_thought"] for frame in observed_frames] == ["3", "4"]
    assert history.frames[0] == HistoryFrame(inputs={"question": "new"})
    assert history.frames[1] == legacy_message
    assert history.frames[-1] == HistoryFrame(outputs={"answer": "done"}, complete=True)


def test_make_truncate_oldest_actions_returns_compaction_fn():
    history = History(frames=[])
    for index in range(4):
        history.append_outputs({"next_thought": str(index)}, observations=[Observation(value=f"result {index}")])

    make_truncate_oldest_actions(max_tokens=0, keep_n=1)(history)

    observed_frames = [frame for frame in history.frames if isinstance(frame, HistoryFrame) and frame.observations]
    assert [frame.outputs["next_thought"] for frame in observed_frames] == ["3"]
