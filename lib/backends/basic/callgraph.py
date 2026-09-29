from ...interfaces.callgraph import Callgraph
from typing import Self


class BasicCallgraph(Callgraph):
    def add(self, data: str) -> Self:
        """# Add Data To Callgraph"""
        if not self.enabled:
            return self
        self.data.append(data)
        return self

    def clear(self) -> Self:
        """# Clear Callgraph"""
        if not self.enabled:
            return self
        self.data = []
        return self

    def save(self, *args, **kwargs) -> Self:
        """# Save Callgraph"""
        if not self.enabled:
            return self
        from json import dumps

        print(dumps(self.data, indent=2))
        return self
