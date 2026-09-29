from ...interfaces.callgraph import Callgraph
from typing import Self


class BasicCallgraph(Callgraph):
    """# Basic Callgraph Backend"""

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

    def save(self: Self, *_args, **_kwargs) -> Self:
        """# Save Callgraph"""
        if not self.enabled:
            return self
        from json import dumps

        print(dumps(self.data, indent=2))
        return self
