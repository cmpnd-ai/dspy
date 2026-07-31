"""Validation and execution-only lowering for Flex module source."""

import ast


def compile_source(source: str) -> tuple[str, str]:
    """Return the module class name and Monty-compatible source."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("module_src must be a non-empty string")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"Invalid module_src: {error}") from error
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1 or any(not isinstance(node, (ast.ClassDef, ast.Pass)) for node in tree.body):
        raise ValueError("module_src must contain exactly one top-level class and no imports or executable statements")
    cls = classes[0]
    if cls.decorator_list or cls.keywords:
        raise ValueError("the module class cannot use decorators or class keywords")
    if len(cls.bases) != 1 or not (
        isinstance(cls.bases[0], ast.Attribute)
        and isinstance(cls.bases[0].value, ast.Name)
        and cls.bases[0].value.id == "dspy"
        and cls.bases[0].attr == "Module"
    ):
        raise ValueError("the top-level class must subclass dspy.Module")
    methods = {node.name: node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    allowed_members = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Pass)
    for index, node in enumerate(cls.body):
        is_docstring = (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        if not isinstance(node, allowed_members) and not is_docstring:
            raise ValueError("the module class can contain only methods and a docstring")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list:
            raise ValueError("module methods cannot use decorators")
    if "__init__" not in methods or "forward" not in methods:
        raise ValueError("the module class must define __init__ and forward")
    if any(isinstance(node, ast.AsyncFunctionDef) for node in ast.walk(cls)):
        raise ValueError("async methods are not supported by MontyProgram")
    cls.bases = []
    init = methods["__init__"]
    for index, statement in enumerate(init.body):
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and not statement.value.args
            and not statement.value.keywords
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "__init__"
            and isinstance(statement.value.func.value, ast.Call)
            and isinstance(statement.value.func.value.func, ast.Name)
            and statement.value.func.value.func.id == "super"
        ):
            init.body[index] = ast.copy_location(ast.Pass(), statement)
    ast.fix_missing_locations(tree)
    return cls.name, ast.unparse(tree)
