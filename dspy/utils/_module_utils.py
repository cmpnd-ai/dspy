import textwrap

import orjson

from dspy.adapters.utils import get_field_description_string


def inspect_modules(program):
    separator = "-" * 80
    output = [separator]

    for name, predictor in program.named_predictors():
        signature = predictor.signature
        instructions = textwrap.dedent(signature.instructions)
        instructions = ("\n" + "\t" * 2).join([""] + instructions.splitlines())

        output.append(f"Module {name}")
        output.append("\n\tInput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + get_field_description_string(signature.input_fields).splitlines()))
        output.append("\tOutput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + get_field_description_string(signature.output_fields).splitlines()))
        output.append(f"\tOriginal Instructions: {instructions}")
        output.append(separator)

    return "\n".join([o.strip("\n") for o in output])


def recursive_mask(o):
    # If the object is already serializable, return it.
    try:
        orjson.dumps(o)
        return o
    except TypeError:
        pass

    # Apply recursively to supported containers, preserving their type.
    if isinstance(o, dict):
        return {k: recursive_mask(v) for k, v in o.items()}
    if isinstance(o, list):
        return [recursive_mask(v) for v in o]
    if isinstance(o, tuple):
        return tuple(recursive_mask(v) for v in o)
    return f"<non-serializable: {type(o).__name__}>"
