from typing import Self, Callable, TypeVar, Generic

T = TypeVar("T")
V = TypeVar("V")


class Maybe(Generic[T]):
    """# Maybe

    Wrapper for nullable data

    Calling the object with a function will return any non-nulled data as input to the function

    Treating the object as a bool will return if it is non-nulled

    `catch` will allow you to return a Maybe with `data` if the called Maybe is null

    `unwrap` will assert that the data is non-nulled and return the internal data
    """

    data: T | None

    def __init__(self: Self, data: T | None = None):
        """# Set the data"""
        self.data = data

    def __call__(self: Self, fptr: Callable[[T], V]) -> "Maybe[V]":
        """# Call a function on non-nulled data"""
        if self.data is None:
            return Maybe(None)
        return Maybe(fptr(self.data))

    def __bool__(self: Self) -> bool:
        """# Is Non-nulled?"""
        return self.data is not None

    def catch(self: Self, other_data: T | None) -> "Self | Maybe[T]":
        """# Non-nulled data returns a new maybe with `other_data`
        Otherwise, return the current Maybe
        """
        if self.data is None:
            return Maybe(other_data)
        return self

    def unwrap(self: Self) -> T:
        """# Get the internal non-nulled data"""
        assert self.data is not None, "Empty container unwraped!"
        return self.data
