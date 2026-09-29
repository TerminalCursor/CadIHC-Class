from ...tools.call_graph import cg
from typing import Self, Callable, Any


class Maybe:
    """# Maybe

    Wrapper for nullable data

    Calling the object with a function will return any non-nulled data as input to the function

    Treating the object as a bool will return if it is non-nulled

    `catch` will allow you to return a Maybe with `data` if the called Maybe is null

    `unwrap` will assert that the data is non-nulled and return the internal data
    """

    data: Any

    def __init__(self: Self, data: Any | None = None):
        """# Set the data"""
        cg.add("Maybe.__init__")
        self.data = data

    def __call__(self: Self, fptr: Callable) -> "Maybe":
        """# Call a function on non-nulled data"""
        cg.add("Maybe.__call__")
        if self.data is None:
            return Maybe(None)
        return Maybe(fptr(self.data))

    def __bool__(self: Self) -> bool:
        """# Is Non-nulled?"""
        cg.add("Maybe.__bool__")
        return self.data is not None

    def catch(self: Self, other_data: Any | None) -> "Maybe":
        """# Non-nulled data returns a new maybe with `other_data`
        Otherwise, return the current Maybe
        """
        cg.add("Maybe.catch")
        if self.data is None:
            return Maybe(other_data)
        return self

    def unwrap(self: Self) -> Any:
        """# Get the internal non-nulled data"""
        cg.add("Maybe.unwrap")
        assert self is not None, "Empty container unwraped!"
        return self.data
