from ...interfaces.callgraph import Callgraph
from typing import Self, List, Any


class BasicCallgraph(Callgraph):
    """# Basic Callgraph Backend"""

    data: List[str] = []
    enabled: bool = False

    def __init__(self: Self):
        """# Initialize Basic Callgraph"""
        super(Callgraph, self).__init__()

    def add(self: Self, data: str) -> Self:
        """# Add Data To Callgraph"""
        if not self.enabled:
            return self
        self.data.append(data)
        return self

    def clear(self: Self) -> Self:
        """# Clear Callgraph"""
        if not self.enabled:
            return self
        self.data = []
        return self

    def save(self: Self, *_args: Any, **_kwargs: Any) -> Self:
        """# Save Callgraph"""
        if not self.enabled:
            return self
        from json import dumps

        print(dumps(self.data, indent=2))
        return self
