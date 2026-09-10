import pytest

from app.core.safe_math import (
    MAX_EXPRESSION_LENGTH,
    UnsafeExpression,
    safe_eval,
)
from app.core.tool_implementations import calculator

# --- Arithmetic it must support ---


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2+2", 4),
        ("10 * 3 + 5", 35),
        ("100 / 4", 25.0),
        ("7 // 2", 3),
        ("7 % 3", 1),
        ("2 ** 10", 1024),
        ("-5 + 3", -2),
        ("+5", 5),
        ("(2 + 3) * 4", 20),
        ("1.5 * 2", 3.0),
        ("((1 + 2) * (3 + 4)) - 5", 16),
    ],
)
def test_valid_expressions(expression, expected):
    assert safe_eval(expression) == expected


# --- The escapes that make bare eval() unusable ---


def test_rejects_builtin_call():
    with pytest.raises(UnsafeExpression):
        safe_eval("__import__('os').system('ls')")


def test_rejects_attribute_walk_out_of_the_sandbox():
    """
    The classic escape from eval(expr, {"__builtins__": {}}): reach any
    object's type, walk up to object, and enumerate every loaded subclass.
    """
    with pytest.raises(UnsafeExpression):
        safe_eval("(1).__class__.__mro__[-1].__subclasses__()")


def test_rejects_name_reference():
    with pytest.raises(UnsafeExpression):
        safe_eval("open")


def test_rejects_function_call():
    with pytest.raises(UnsafeExpression):
        safe_eval("print(1)")


def test_rejects_subscript():
    with pytest.raises(UnsafeExpression):
        safe_eval("[1, 2, 3][0]")


def test_rejects_comprehension():
    with pytest.raises(UnsafeExpression):
        safe_eval("[x for x in range(10)]")


def test_rejects_lambda():
    with pytest.raises(UnsafeExpression):
        safe_eval("(lambda: 1)()")


def test_rejects_string_literal():
    with pytest.raises(UnsafeExpression):
        safe_eval("'hello'")


def test_rejects_boolean():
    with pytest.raises(UnsafeExpression):
        safe_eval("True")


def test_rejects_comparison():
    with pytest.raises(UnsafeExpression):
        safe_eval("1 < 2")


def test_rejects_walrus_assignment():
    with pytest.raises(UnsafeExpression):
        safe_eval("(x := 5)")


# --- Resource limits ---


def test_rejects_oversized_expression():
    with pytest.raises(UnsafeExpression, match="exceeds"):
        safe_eval("1+" * MAX_EXPRESSION_LENGTH + "1")


def test_rejects_huge_exponent():
    """2 ** (2 ** 30) would hang the worker, not raise."""
    with pytest.raises(UnsafeExpression, match="exponent"):
        safe_eval("2 ** 999999")


def test_rejects_empty_expression():
    with pytest.raises(UnsafeExpression):
        safe_eval("")


def test_rejects_syntax_error():
    with pytest.raises(UnsafeExpression, match="could not parse"):
        safe_eval("2 +")


def test_division_by_zero_raises_arithmetic_error():
    with pytest.raises(ZeroDivisionError):
        safe_eval("1 / 0")


# --- The tool wrapper turns all of it into a message for the model ---


def test_calculator_returns_result_as_string():
    assert calculator("10 * 3 + 5") == "35"


def test_calculator_reports_unsafe_expression_as_error():
    result = calculator("__import__('os').system('ls')")

    assert result.startswith("ERROR")


def test_calculator_reports_division_by_zero_as_error():
    result = calculator("1 / 0")

    assert result.startswith("ERROR")
    assert "zero" in result


def test_calculator_never_raises():
    """
    The execution loop relies on tools returning a string it can feed back
    to the model — a tool that raises would abort the whole run.
    """
    for expression in ("", "open", "2 +", "1/0", "2**999999", "'x'"):
        assert calculator(expression).startswith("ERROR")
