import pytest
from histclass.interfaces.monad.maybe import Maybe

# pyright: reportUnknownVariableType=false,reportUnknownMemberType=false


def test_maybe_init_with_none():
    """Test Maybe initialization with None."""
    maybe = Maybe(None)
    assert maybe.data is None


def test_maybe_init_with_value():
    """Test Maybe initialization with a value."""
    maybe = Maybe(42)
    assert maybe.data == 42


def test_maybe_call_with_none_data():
    """Test calling function on None data."""

    def double(x: int) -> int:
        return x * 2

    maybe = Maybe(None)
    result = maybe(double)
    assert result.data is None


def test_maybe_call_with_value():
    """Test calling function on non-None data."""

    def double(x: int) -> int:
        return x * 2

    maybe = Maybe(5)
    result = maybe(double)
    assert result.data == 10


def test_maybe_bool_false():
    """Test boolean conversion with None data."""
    maybe = Maybe(None)
    assert bool(maybe) is False  # pyright: ignore


def test_maybe_bool_true():
    """Test boolean conversion with non-None data."""
    maybe = Maybe(42)
    assert bool(maybe) is True


def test_maybe_catch_with_none_data():
    """Test catch method when current data is None."""
    maybe = Maybe(None)
    result = maybe.catch(100)
    assert result.data == 100


def test_maybe_catch_with_value_data():
    """Test catch method when current data is not None."""
    maybe = Maybe(42)
    result = maybe.catch(100)
    assert result.data == 42


def test_maybe_unwrap_with_none_data():
    """Test unwrap method with None data should raise AssertionError."""
    maybe = Maybe(None)
    with pytest.raises(AssertionError, match="Cannot unwrap a null Maybe instance!"):
        maybe.unwrap()


def test_maybe_unwrap_with_value_data():
    """Test unwrap method with non-None data."""
    maybe = Maybe(42)
    result = maybe.unwrap()
    assert result == 42


def test_maybe_call_composition():
    """Test calling multiple functions in sequence."""

    def add_one(x: int) -> int:
        return x + 1

    def multiply_by_two(x: int) -> int:
        return x * 2

    maybe = Maybe(5)
    result = maybe(add_one)(multiply_by_two)
    assert result.data == 12  # (5 + 1) * 2 = 12
