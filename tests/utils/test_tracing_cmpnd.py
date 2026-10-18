"""Executable local consumer contract, not cmpnd SDK/backend conformance."""

import hashlib
import importlib.util
import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from litellm import Choices, Message, ModelResponse

import dspy
from dspy.utils.tracing import InvocationRef, capture_trace

spec = importlib.util.spec_from_file_location(
    "tracing_cmpnd_example", Path(__file__).parents[2] / "docs/examples/tracing_cmpnd.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)
CmpndObserver = example.CmpndObserver


@pytest.fixture
def consumer():
    events = []
    return CmpndObserver(lambda path, body: events.append((path, body))), events


@pytest.fixture
def provider(monkeypatch):
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return ModelResponse(
            id="provider-response",
            model="reported-model",
            service_tier="default",
            choices=[Choices(message=Message(content="[[ ## answer ## ]]\nyes\n[[ ## completed ## ]]"))],
            usage={
                "prompt_tokens": 7,
                "completion_tokens": 0,
                "total_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        )

    monkeypatch.setattr("litellm.completion", complete)
    return calls


def spans(events):
    return [span for path, body in events if path.endswith("/spans") for span in body["spans"]]


def test_repeated_lm_without_capture_or_history_preserves_cache_evidence(consumer, provider):
    observer, events = consumer
    lm = dspy.LM("openai/test-model", engine="litellm", cache=True, service_tier="auto")
    prompt = f"unique-{uuid4()}"
    with dspy.context(execution_observers=[observer], disable_history=True):
        lm(prompt)
        lm(prompt)
    observer.finalize()
    first, cached = spans(events)
    assert len(provider) == 1
    assert lm.history == []
    assert first["span_id"] != cached["span_id"]
    evidence = first["lm_usage"]
    assert evidence["usage_raw"]["prompt_tokens_details"]["cached_tokens"] == 2
    assert evidence["model_requested"] == "openai/test-model"
    assert evidence["model_reported"] == "reported-model"
    assert evidence["service_tier_requested"] == "auto"
    assert evidence["service_tier_effective"] == "default"
    assert evidence["completion_tokens"] == 0
    assert "reasoning_tokens" not in evidence  # absent is not a reported zero
    assert cached["lm_usage"]["dspy_client_cache_hit"] is True
    assert cached["lm_usage"]["usage_reported"] is False
    assert cached["total_tokens"] is None
    # Legacy cache currently strips provider usage; the consumer must not invent it.
    assert cached["lm_usage"]["usage_raw"] == json.loads(cached["attributes"]["dspy.lm"])["usage_raw"]
    assert "provider_cost" not in evidence
    preserved = json.loads(first["attributes"]["dspy.lm"])
    assert preserved["response_ids"] == ["provider-response"]
    assert preserved["finish_reasons"] == ["stop"]
    unsigned = {k: v for k, v in evidence.items() if k != "source_payload_digest"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert evidence["source_payload_digest"] == hashlib.sha256(canonical.encode()).hexdigest()
    json.dumps(events)  # actual in-memory bodies are JSON serializable


def test_parallel_same_lm_streams_children_before_root_end(consumer, provider):
    observer, events = consumer
    lm = dspy.LM("openai/test-model", engine="litellm", cache=False)

    class Program(dspy.Module):
        def forward(self):
            result = dspy.Parallel(num_threads=2, disable_progress_bar=True)(
                [(lm, {"prompt": "same"}), (lm, {"prompt": "same"})]
            )
            assert len([s for s in spans(events) if s.get("lm_usage")]) == 2
            assert not any(path.endswith("/finalize") for path, _ in events)
            with pytest.raises(RuntimeError, match="Join"):
                observer.finalize()
            return result

    with dspy.context(execution_observers=[observer], disable_history=True):
        Program()()
    observer.finalize()
    children = [s for s in spans(events) if s.get("lm_usage")]
    root = spans(events)[-1]
    assert len(provider) == 2
    assert len({s["span_id"] for s in children}) == 2
    assert all(s["parent_span_id"] == root["span_id"] for s in children)
    assert all(s["trace_id"] == root["trace_id"] for s in children)
    assert events[-1][0].endswith("/finalize")
    assert UUID(root["trace_id"]).version == 4


@pytest.mark.parametrize("retain", [False, True])
def test_evaluation_candidate_judge_refs_and_failed_attempts(consumer, provider, retain):
    observer, events = consumer
    lm = dspy.LM("openai/test-model", engine="litellm", cache=False)

    class Program(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict("question -> answer")

        def forward(self, question):
            prediction = self.predict(question=question)
            if question == "bad":
                raise ValueError("failed after paid work")
            return prediction

    judge = dspy.Predict("question -> answer")

    def metric(example, prediction):
        return float(judge(question=example.question).answer == prediction.answer)

    evaluator = dspy.Evaluate(
        devset=[dspy.Example(question=q).with_inputs("question") for q in ("ok", "bad")],
        metric=metric,
        num_threads=2,
        max_errors=10,
        capture_execution=retain,
    )
    with dspy.context(lm=lm, execution_observers=[observer], disable_history=True):
        result = evaluator(Program())
    observer.finalize()
    good, failed = result.executions
    assert good.accepted in good.attempts
    attempt = good.accepted
    assert bool(attempt.execution) == retain
    assert isinstance(attempt.program, tuple) and isinstance(attempt.metric, tuple)
    assert attempt.program and attempt.metric
    assert set(attempt.program).isdisjoint(attempt.metric)
    by_ref = {(s["trace_id"], s["span_id"]): s for s in spans(events)}
    for ref in (*attempt.program, *attempt.metric):
        address = observer.address(ref)
        assert (address["trace_id"], address["span_id"]) in by_ref

    # Candidate and judge can share the exact same LM object; provenance,
    # not a model-name/history heuristic, distinguishes their paid calls.
    def lm_descendants(refs):
        roots = {tuple(observer.address(ref).values()) for ref in refs}
        found = set(roots)
        while True:
            expanded = found | {
                key for key, span in by_ref.items() if (span["trace_id"], span["parent_span_id"]) in found
            }
            if expanded == found:
                break
            found = expanded
        return {key for key in found if by_ref[key].get("lm_usage")}

    candidate_calls = lm_descendants(attempt.program)
    judge_calls = lm_descendants(attempt.metric)
    assert len(candidate_calls) == len(judge_calls) == 1
    assert candidate_calls.isdisjoint(judge_calls)
    assert failed.accepted is None
    assert failed.attempts[0].status == "error"
    assert failed.attempts[0].program
    assert failed.attempts[0].execution is None or retain
    failed_ref = observer.address(failed.attempts[0].program[0])
    assert by_ref[(failed_ref["trace_id"], failed_ref["span_id"])]["error_type"] == "ValueError"
    assert any(s["signature_input_fields"] == ["question"] for s in spans(events))
    assert lm.history == []
    assert any(body["status"] == "error" for path, body in events if path.endswith("/finalize"))


def test_external_trace_identity_is_mapped_not_uuid_reformatted(consumer, provider):
    observer, events = consumer
    external = InvocationRef("0123456789abcdef0123456789abcdef", "0123456789abcdef")
    lm = dspy.LM("openai/test-model", engine="litellm", cache=False)
    with dspy.context(execution_observers=[observer], execution_parent=external, disable_history=True):
        with capture_trace() as execution:
            lm("hello")
    observer.finalize()
    (span,) = spans(events)
    assert UUID(span["trace_id"]).version == 4
    assert span["trace_id"].replace("-", "") != external.trace_id
    assert span["parent_span_id"] == external.span_id
    assert json.loads(span["attributes"]["dspy.ref"])["trace_id"] == external.trace_id
    assert execution.invocations[0].component is lm  # detached delivery didn't touch capture
    assert observer.address(execution.invocations[0].ref)["trace_id"] == span["trace_id"]


def test_failed_lm_preserves_partial_accounting_events(consumer):
    from dspy.lm15 import Message as LMMessage
    from dspy.lm15 import Response, Usage

    class Engine:
        calls = 0

        def complete(self, request):
            self.calls += 1
            if self.calls == 2:
                raise dspy.LMAuthError("failed second candidate")
            return Response(
                id="paid",
                model=request.model,
                finish_reason="stop",
                message=LMMessage.assistant("first"),
                usage=Usage(input_tokens=7, output_tokens=3),
            )

    observer, events = consumer
    lm = dspy.LM("custom/accounting", engine=Engine(), cache=False, num_retries=0)
    with dspy.context(execution_observers=[observer], disable_history=True):
        with pytest.raises(dspy.LMAuthError):
            lm("hello", n=2)
    observer.finalize()
    (span,) = spans(events)
    assert span["status"] == "error" and span["span_type"] == "lm_call"
    assert "lm_usage" not in span  # no fabricated whole-call result
    (entry,) = json.loads(span["attributes"]["dspy.data"])["usage"]
    assert entry["model"] == lm.model
    assert entry["usage"]["total_tokens"] == 10
