from ...tools.call_graph import cg
from typing import Callable, Any


class Maybe:
    """# Maybe

    Wrapper for nullable data

    Calling the object with a function will return any non-nulled data as input to the function

    Treating the object as a bool will return if it is non-nulled

    `catch` will allow you to return a Maybe with `data` if the called Maybe is null

    `unwrap` will assert that the data is non-nulled and return the internal data
    """

    def __init__(self, data: Any | None = None):
        """# Set the data"""
        cg.add("Maybe.__init__")
        self.data = data

    def __call__(self, fptr: Callable):
        """# Call a function on non-nulled data"""
        cg.add("Maybe.__call__")
        if self.data is None:
            return Maybe(None)
        return Maybe(fptr(self.data))

    def catch(self, other_data: Any | None):
        """# Non-nulled data returns a new maybe with `other_data`
        Otherwise, return the current Maybe
        """
        cg.add("Maybe.catch")
        if self.data is None:
            return Maybe(other_data)
        return self

    def __bool__(self):
        """# Is Non-nulled?"""
        cg.add("Maybe.__bool__")
        return self.data is not None

    def unwrap(self):
        """# Get the internal non-nulled data"""
        cg.add("Maybe.unwrap")
        assert self.data is not None
        return self.data
