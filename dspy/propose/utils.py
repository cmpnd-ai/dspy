import ast
import inspect
import linecache
import re

import dspy
from dspy.predict.parameter import Parameter
from dspy.teleprompt.utils import get_signature


def strip_prefix(text):
    pattern = r"^[\*\s]*(([\w\'\-]+\s+){0,4}[\w\'\-]+):\s*"
    modified_text = re.sub(pattern, "", text)
    return modified_text.strip('"')


def create_predictor_level_history_string(base_program, predictor_i, trial_logs, top_n):
    instruction_aggregate = {}
    for trial in trial_logs.values():
        if "program_path" not in trial:
            continue
        trial_program = base_program.deepcopy()
        trial_program.load(trial["program_path"])
        predictor = trial_program.predictors()[predictor_i]
        instruction = get_signature(predictor).instructions
        score = trial["score"]
        if instruction in instruction_aggregate:
            instruction_aggregate[instruction]["total_score"] += score
            instruction_aggregate[instruction]["count"] += 1
        else:
            instruction_aggregate[instruction] = {"total_score": score, "count": 1}

    averages = [
        (instruction, data["total_score"] / data["count"]) for instruction, data in instruction_aggregate.items()
    ]
    top_instructions = sorted(averages, key=lambda item: item[1], reverse=True)[:top_n]
    return "".join(instruction + f" | Score: {score}\n\n" for instruction, score in reversed(top_instructions))


def create_example_string(fields, example):
    return "\n".join(f"{field.json_schema_extra['prefix']} {example.get(name)}" for name, field in fields.items())


def get_dspy_source_code(module):
    header = []
    base_code = ""

    # Don't get source code for Predict or ChainOfThought modules (NOTE we will need to extend this list as more DSPy.modules are added)
    # TODO: if type(module).__name__ not in ["Predict", "ChainOfThought", "ReAct"]:
    if not type(module).__name__ == "Predict" and not type(module).__name__ == "ChainOfThought":
        try:
            base_code = inspect.getsource(type(module))
        except (TypeError, OSError):
            obj = type(module)
            # Notebook classes have no module file; an own method identifies
            # both the cell and the correct class when names are reused.
            method = next(
                (
                    member
                    for _, member in inspect.getmembers(obj)
                    if inspect.isfunction(member) and member.__qualname__ == f"{obj.__qualname__}.{member.__name__}"
                ),
                None,
            )
            if method is None:
                raise TypeError(f"Source for {obj!r} not found")
            lines = linecache.getlines(inspect.getfile(method))
            for node in ast.walk(ast.parse("".join(lines))):
                if (
                    isinstance(node, ast.ClassDef)
                    and node.name == obj.__name__
                    and node.lineno <= method.__code__.co_firstlineno <= node.end_lineno
                ):
                    start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
                    base_code = "".join(lines[start - 1 : node.end_lineno])
                    break
            else:
                raise OSError(f"Source for {obj!r} not found")

    completed_set = set()
    for attribute in module.__dict__.keys():
        try:
            iterable = iter(getattr(module, attribute))
        except TypeError:
            iterable = [getattr(module, attribute)]

        for item in iterable:
            # Skip items that are unhashable (like module history)
            try:
                hash(item)
            except TypeError:
                continue
            if isinstance(item, Parameter):
                if (
                    hasattr(item, "signature")
                    and item.signature is not None
                    and item.signature.__pydantic_parent_namespace__["signature_name"] + "_sig" not in completed_set
                ):
                    try:
                        header.append(inspect.getsource(item.signature))
                        print(inspect.getsource(item.signature))
                    except (TypeError, OSError):
                        header.append(str(item.signature))
                    completed_set.add(item.signature.__pydantic_parent_namespace__["signature_name"] + "_sig")
            if isinstance(item, dspy.Module):
                code = get_dspy_source_code(item).strip()
                if code not in completed_set:
                    header.append(code)
                    completed_set.add(code)
            completed_set.add(item)

    return "\n\n".join(header) + "\n\n" + base_code
