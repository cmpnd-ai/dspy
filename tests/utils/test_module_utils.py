import orjson

import dspy
from dspy.predict.refine import inspect_modules as refine_inspect_modules
from dspy.predict.refine import recursive_mask as refine_recursive_mask
from dspy.teleprompt.simba_utils import inspect_modules as simba_inspect_modules
from dspy.teleprompt.simba_utils import recursive_mask as simba_recursive_mask


class NonSerializable:
    pass


def test_inspect_modules_preserves_output_and_public_imports():
    program = dspy.Predict("question -> answer")
    expected = (
        "--------------------------------------------------------------------------------\n"
        "Module self\n"
        "\tInput Fields:\n\t\t1. `question` (str):\n"
        "\tOutput Fields:\n\t\t1. `answer` (str):\n"
        "\tOriginal Instructions: \n"
        "\t\tGiven the fields `question`, produce the fields `answer`.\n"
        "--------------------------------------------------------------------------------"
    )

    assert refine_inspect_modules(program) == expected
    assert simba_inspect_modules(program) == expected


def test_recursive_mask_preserves_serializable_objects_by_identity():
    value = {"nested": [None, True, 0, 1.5, "text", ("tuple",)]}

    assert refine_recursive_mask(value) is value
    assert simba_recursive_mask(value) is value


def test_recursive_mask_masks_nested_edge_values_and_preserves_containers():
    value = {
        "dict": {"value": NonSerializable()},
        "list": [NonSerializable()],
        "tuple": (NonSerializable(),),
    }
    expected = {
        "dict": {"value": "<non-serializable: NonSerializable>"},
        "list": ["<non-serializable: NonSerializable>"],
        "tuple": ("<non-serializable: NonSerializable>",),
    }

    assert refine_recursive_mask(value) == expected
    assert simba_recursive_mask(value) == expected
    assert orjson.loads(orjson.dumps(expected)) == {
        **expected,
        "tuple": ["<non-serializable: NonSerializable>"],
    }


def test_recursive_mask_preserves_non_string_dictionary_keys_for_final_dump_fallback():
    value = {(1, 2): NonSerializable()}
    expected = {(1, 2): "<non-serializable: NonSerializable>"}

    assert refine_recursive_mask(value) == expected
    assert simba_recursive_mask(value) == expected
