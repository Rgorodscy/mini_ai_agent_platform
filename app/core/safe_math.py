"""
A restricted arithmetic evaluator for the calculator tool.

Tool arguments are written by the LLM, which in turn can be steered by
untrusted input (a user task, or a document pulled in by RAG). `eval` is
therefore unusable here, with or without a stripped `__builtins__`:
`(1).__class__.__mro__[-1].__subclasses__()` walks out of any such
sandbox and reaches the import machinery.

Instead the expression is parsed to an AST and only an explicit allowlist
of arithmetic nodes is interpreted. Anything else — a name, a call, an
attribute access, a subscript — is rejected before evaluation.
"""

import ast
import operator

MAX_EXPRESSION_LENGTH = 200

# Bounds the cost of a single expression: 2 ** (2 ** 30) would otherwise
# hang the worker with no syntax error in sight.
MAX_EXPONENT = 100

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class UnsafeExpression(ValueError):
    """Raised when an expression contains anything but plain arithmetic."""


def safe_eval(expression: str) -> float | int:
    """
    Evaluates an arithmetic expression.

    Raises UnsafeExpression for anything outside the allowlist, and the
    usual arithmetic errors (ZeroDivisionError, OverflowError) for input
    that is well-formed but cannot be computed.
    """
    if not expression or not expression.strip():
        raise UnsafeExpression("empty expression")

    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise UnsafeExpression(
            f"expression exceeds {MAX_EXPRESSION_LENGTH} characters"
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpression(f"could not parse expression: {e.msg}") from e

    return _evaluate(tree.body)


def _evaluate(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value, (int, float)
        ):
            raise UnsafeExpression(
                f"only numbers are allowed, got {type(node.value).__name__}"
            )
        return node.value

    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(
                f"operator not allowed: {type(node.op).__name__}"
            )

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        if op is operator.pow and abs(right) > MAX_EXPONENT:
            raise UnsafeExpression(
                f"exponent exceeds the limit of {MAX_EXPONENT}"
            )

        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(
                f"operator not allowed: {type(node.op).__name__}"
            )
        return op(_evaluate(node.operand))

    raise UnsafeExpression(f"not allowed: {type(node).__name__}")
