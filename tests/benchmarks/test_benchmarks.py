"""Performance benchmarks for DSPy core operations.

These benchmarks cover the most performance-critical, CPU-bound code paths
in DSPy: data container operations, signature parsing and manipulation,
adapter formatting and parsing, and serialization.
"""

import copy
import json

import pytest

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    get_field_description_string,
    parse_value,
    serialize_for_json,
    translate_field_type,
)
from dspy.primitives.example import Example
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import Signature, infer_prefix, make_signature

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_example():
    return Example(question="What is the capital of France?", answer="Paris")


@pytest.fixture
def large_example():
    return Example(
        **{f"field_{i}": f"value_{i}" for i in range(50)},
    )


@pytest.fixture
def nested_example():
    inner = Example(detail="nested_value", score=42)
    return Example(
        question="What is DSPy?",
        answer="A framework",
        context=["paragraph one", "paragraph two", "paragraph three"],
        metadata={"source": "test", "nested": inner},
    )


@pytest.fixture
def simple_signature():
    return make_signature("question -> answer")


@pytest.fixture
def complex_signature():
    return make_signature("question, context: list[str], hint -> answer, reasoning")


@pytest.fixture
def chat_adapter():
    return ChatAdapter()


@pytest.fixture
def qa_signature_class():
    class QA(Signature):
        """Answer the question based on the context."""

        question: str = dspy.InputField(desc="The question to answer")
        context: str = dspy.InputField(desc="Relevant context")
        answer: str = dspy.OutputField(desc="The answer")

    return QA


@pytest.fixture
def chat_completion_text(qa_signature_class):
    return "[[ ## answer ## ]]\nParis is the capital of France.\n\n[[ ## completed ## ]]"


# ---------------------------------------------------------------------------
# Example / Prediction benchmarks
# ---------------------------------------------------------------------------


class TestExampleBenchmarks:
    def test_example_creation(self, benchmark):
        """Benchmark creating an Example from keyword arguments."""
        benchmark(Example, question="What is 2+2?", answer="4")

    def test_example_creation_from_dict(self, benchmark):
        """Benchmark creating an Example from a base dictionary."""
        data = {f"field_{i}": f"value_{i}" for i in range(20)}
        benchmark(Example, base=data)

    def test_example_field_access(self, benchmark, simple_example):
        """Benchmark attribute-style field access."""

        def access_fields():
            _ = simple_example.question
            _ = simple_example.answer

        benchmark(access_fields)

    def test_example_copy(self, benchmark, simple_example):
        """Benchmark shallow copy with field override."""
        benchmark(simple_example.copy, answer="London")

    def test_example_with_inputs(self, benchmark, simple_example):
        """Benchmark marking input fields."""
        benchmark(simple_example.with_inputs, "question")

    def test_example_inputs_labels(self, benchmark):
        """Benchmark splitting an Example into inputs and labels."""
        ex = Example(question="Why?", answer="Because.", context="info").with_inputs("question")

        def split():
            _ = ex.inputs()
            _ = ex.labels()

        benchmark(split)

    def test_example_to_dict(self, benchmark, nested_example):
        """Benchmark recursive serialization to dict."""
        benchmark(nested_example.toDict)

    def test_example_hash(self, benchmark, simple_example):
        """Benchmark hash computation."""
        benchmark(hash, simple_example)

    def test_example_keys_values_items(self, benchmark, large_example):
        """Benchmark dict-like iteration over a large Example."""

        def iterate():
            _ = large_example.keys()
            _ = large_example.values()
            _ = large_example.items()

        benchmark(iterate)

    def test_prediction_creation(self, benchmark):
        """Benchmark creating a Prediction."""
        benchmark(Prediction, answer="Paris", score=0.95)

    def test_prediction_arithmetic(self, benchmark):
        """Benchmark Prediction score arithmetic."""
        p = Prediction(answer="Paris", score=0.8)

        def arithmetic():
            _ = p + p
            _ = p / 2

        benchmark(arithmetic)


# ---------------------------------------------------------------------------
# Signature benchmarks
# ---------------------------------------------------------------------------


class TestSignatureBenchmarks:
    def test_make_signature_simple(self, benchmark):
        """Benchmark parsing a simple signature string."""
        benchmark(make_signature, "question -> answer")

    def test_make_signature_complex(self, benchmark):
        """Benchmark parsing a complex signature with typed fields."""
        benchmark(make_signature, "question: str, context: list[str], hint: str -> answer: str, confidence: float")

    def test_signature_with_instructions(self, benchmark, simple_signature):
        """Benchmark creating a signature with new instructions."""
        benchmark(simple_signature.with_instructions, "Translate the question to French.")

    def test_signature_append_field(self, benchmark, simple_signature):
        """Benchmark appending an output field."""
        benchmark(simple_signature.append, "confidence", dspy.OutputField(desc="Confidence score"), float)

    def test_signature_prepend_field(self, benchmark, simple_signature):
        """Benchmark prepending an input field."""
        benchmark(simple_signature.prepend, "context", dspy.InputField(desc="Context"))

    def test_signature_delete_field(self, benchmark, complex_signature):
        """Benchmark deleting a field from a signature."""
        benchmark(complex_signature.delete, "hint")

    def test_signature_dump_state(self, benchmark, complex_signature):
        """Benchmark serializing signature state."""
        benchmark(complex_signature.dump_state)

    def test_signature_load_state(self, benchmark, complex_signature):
        """Benchmark deserializing signature state."""
        state = complex_signature.dump_state()
        benchmark(complex_signature.load_state, state)

    def test_signature_equals(self, benchmark, simple_signature):
        """Benchmark comparing two signatures."""
        other = make_signature("question -> answer")
        benchmark(simple_signature.equals, other)

    def test_infer_prefix_camel_case(self, benchmark):
        """Benchmark prefix inference from camelCase."""
        benchmark(infer_prefix, "camelCaseFieldName")

    def test_infer_prefix_snake_case(self, benchmark):
        """Benchmark prefix inference from snake_case."""
        benchmark(infer_prefix, "snake_case_field_name")


# ---------------------------------------------------------------------------
# Adapter formatting / parsing benchmarks
# ---------------------------------------------------------------------------


class TestAdapterBenchmarks:
    def test_chat_adapter_format_field_description(self, benchmark, chat_adapter, qa_signature_class):
        """Benchmark formatting field descriptions for a signature."""
        benchmark(chat_adapter.format_field_description, qa_signature_class)

    def test_chat_adapter_format_field_structure(self, benchmark, chat_adapter, qa_signature_class):
        """Benchmark formatting the field structure section."""
        benchmark(chat_adapter.format_field_structure, qa_signature_class)

    def test_chat_adapter_format_user_message(self, benchmark, chat_adapter, qa_signature_class):
        """Benchmark formatting a user message from inputs."""
        inputs = {"question": "What is the capital of France?", "context": "France is a country in Europe."}
        benchmark(chat_adapter.format_user_message_content, qa_signature_class, inputs)

    def test_chat_adapter_format_assistant_message(self, benchmark, chat_adapter, qa_signature_class):
        """Benchmark formatting an assistant response message."""
        outputs = {"answer": "Paris is the capital of France."}
        benchmark(chat_adapter.format_assistant_message_content, qa_signature_class, outputs)

    def test_chat_adapter_parse(self, benchmark, chat_adapter):
        """Benchmark parsing a completion into structured fields."""
        sig = make_signature("question -> answer")
        completion = "[[ ## answer ## ]]\nParis is the capital of France.\n\n[[ ## completed ## ]]"
        benchmark(chat_adapter.parse, sig, completion)

    def test_chat_adapter_parse_multifield(self, benchmark, chat_adapter):
        """Benchmark parsing a multi-field completion."""
        sig = make_signature("question -> answer: str, reasoning: str, confidence: float")
        completion = (
            "[[ ## answer ## ]]\nParis\n\n"
            "[[ ## reasoning ## ]]\nFrance's capital is Paris.\n\n"
            "[[ ## confidence ## ]]\n0.95\n\n"
            "[[ ## completed ## ]]"
        )
        benchmark(chat_adapter.parse, sig, completion)

    def test_format_field_value_string(self, benchmark):
        """Benchmark formatting a simple string field value."""
        sig = make_signature("question -> answer")
        field_info = sig.fields["answer"]
        benchmark(format_field_value, field_info, "Paris is the capital of France.")

    def test_format_field_value_list(self, benchmark):
        """Benchmark formatting a list field value."""
        sig = make_signature("question -> answer")
        field_info = sig.fields["question"]
        value = [f"Paragraph {i}: Some context about the topic." for i in range(10)]
        benchmark(format_field_value, field_info, value)

    def test_serialize_for_json_complex(self, benchmark):
        """Benchmark JSON serialization of a complex nested structure."""
        data = {
            "results": [{"text": f"result {i}", "score": 0.9 - i * 0.1} for i in range(10)],
            "metadata": {"source": "test", "count": 10},
        }
        benchmark(serialize_for_json, data)

    def test_translate_field_type_string(self, benchmark, qa_signature_class):
        """Benchmark field type translation for a string field."""
        field_info = qa_signature_class.fields["answer"]
        benchmark(translate_field_type, "answer", field_info)

    def test_parse_value_string(self, benchmark):
        """Benchmark parsing a string value."""
        benchmark(parse_value, "Paris", str)

    def test_parse_value_int(self, benchmark):
        """Benchmark parsing an integer value from a string."""
        benchmark(parse_value, "42", int)

    def test_parse_value_list(self, benchmark):
        """Benchmark parsing a JSON list value."""
        benchmark(parse_value, '["a", "b", "c"]', list[str])

    def test_get_annotation_name_simple(self, benchmark):
        """Benchmark getting the name of a simple type annotation."""
        benchmark(get_annotation_name, str)

    def test_get_annotation_name_generic(self, benchmark):
        """Benchmark getting the name of a generic type annotation."""
        benchmark(get_annotation_name, list[str])

    def test_get_field_description_string(self, benchmark, qa_signature_class):
        """Benchmark generating field description strings."""
        benchmark(get_field_description_string, qa_signature_class.output_fields)


# ---------------------------------------------------------------------------
# Module state serialization benchmarks
# ---------------------------------------------------------------------------


class TestSerializationBenchmarks:
    def test_example_deepcopy(self, benchmark, nested_example):
        """Benchmark deep copying a nested Example."""
        benchmark(copy.deepcopy, nested_example)

    def test_signature_deepcopy(self, benchmark, complex_signature):
        """Benchmark deep copying a complex signature's fields."""
        benchmark(copy.deepcopy, complex_signature.fields)

    def test_example_json_roundtrip(self, benchmark, nested_example):
        """Benchmark JSON serialization and deserialization of an Example."""

        def roundtrip():
            d = nested_example.toDict()
            _ = json.dumps(d)

        benchmark(roundtrip)
