from typing import Self, Callable, TypeVar, Generic

T = TypeVar("T")
V = TypeVar("V")


class Maybe(Generic[T]):
    """
    A wrapper class for nullable data.

    This class provides a way to handle optional values in a type-safe manner.
    It allows you to call functions on the non-nulled data and return a new
    `Maybe` instance with the result if the original data is null. You can also
    use it as a boolean value, where `False` represents null data.

    Attributes:
        data (T | None): The internal data or None if it's null.
    """

    data: T | None

    def __init__(self: Self, data: T | None = None):
        """
        Initialize the Maybe instance with the given data.

        Args:
            data (T | None): The initial data. Defaults to None.
        """
        self.data = data

    def __call__(self: Self, fptr: Callable[[T], V]) -> "Maybe[V]":
        """
        Call a function on the non-nulled data and return a new Maybe instance
        with the result.

        Args:
            fptr (Callable[[T], V]): The function to call on the data.

        Returns:
            Maybe[V]: A new Maybe instance with the result.
        """
        if self.data is None:
            return Maybe(None)
        return Maybe(fptr(self.data))

    def __bool__(self: Self) -> bool:
        """
        Check if the data is non-nulled.

        Returns:
            bool: True if the data is not None, False otherwise.
        """
        return self.data is not None

    def catch(self: Self, other_data: T | None) -> "Self | Maybe[T]":
        """
        Return a new Maybe instance with the given data if this instance is null,
        or return the current instance.

        Args:
            other_data (T | None): The alternative data to use.

        Returns:
            Self | Maybe[T]: A new Maybe instance with the result, or the current
            instance.
        """
        if self.data is None:
            return Maybe(other_data)
        return self

    def unwrap(self: Self) -> T:
        """
        Get the internal non-nulled data and raise an assertion if it's null.

        Returns:
            T: The non-nulled data.
        """
        assert self.data is not None, "Cannot unwrap a null Maybe instance!"
        return self.data
